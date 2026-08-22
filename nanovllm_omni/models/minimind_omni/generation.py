"""Project-owned optimized MiniMind-O generation loop.

This intentionally keeps the vendored model untouched.  The model's forward
method remains the upstream implementation; only the token/audio sampling
loop and its growing input buffers live here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def _sample_audio_codes(
    audio_logits: list[Any], audio_codes: list[list[int]], active: list[int]
) -> dict[int, int]:
    """Sample active audio streams as one GPU batch and sync once."""
    import torch
    import torch.nn.functional as functional

    logits = torch.stack([audio_logits[i][0, -1, :].clone() / 0.2 for i in active])
    for row, layer in enumerate(active):
        for previous in audio_codes[layer][-3:]:
            logits[row, previous] /= 1.05
    top_values, top_indices = logits.topk(50, dim=-1)
    probabilities = functional.softmax(top_values, dim=-1)
    # Keep one multinomial call per stream: batching this call changes the
    # Philox draw order and therefore changes subsequent text samples.
    sampled = torch.cat([torch.multinomial(probabilities[row], 1) for row in range(len(active))])
    codes = top_indices.gather(1, sampled[:, None]).flatten().tolist()
    return dict(zip(active, codes, strict=True))


def stream_generate_optimized(
    model: Any,
    input_ids: Any,
    eos_token_id: int | None = 2,
    max_new_tokens: int = 1024,
    temperature: float = 0.75,
    top_p: float = 0.90,
    rp: float = 1.0,
    use_cache: bool = True,
    return_audio_codes: bool = False,
    **args: Any,
) -> Iterator[tuple[Any, Any]]:
    """Stream MiniMind-O output with project-owned low-sync bookkeeping.

    The text sample still synchronizes once per step because the next model
    input depends on it.  Eight audio samples are kept on device and copied to
    Python together in one transfer.  Input/audio history buffers are fixed
    capacity, avoiding repeated growth ``torch.cat`` calls in this loop.
    """
    import torch
    import torch.nn.functional as functional

    start_pos = input_ids.shape[1]
    capacity = start_pos + max_new_tokens
    past_kvs, text_finished, first_finished = None, False, True
    audio_codes: list[list[int]] = [[] for _ in range(8)]
    audio_stop_pos: list[int | None] = [None] * 8
    audio_pad = model.audio_pad_token
    audio_stop = model.audio_stop_token
    audio_spk = model.audio_spk_token
    audio_buffer = torch.full(
        (1, 8, capacity), audio_pad, dtype=torch.long, device=input_ids.device
    )
    text_buffer = torch.empty((1, capacity), dtype=input_ids.dtype, device=input_ids.device)
    text_buffer[:, :start_pos] = input_ids
    audio_input = torch.empty((1, 9, 1), dtype=torch.long, device=input_ids.device)

    spk_emb = args.get("spk_emb")
    ref_codes = args.get("ref_codes")
    ref_len = ref_codes.shape[2] if ref_codes is not None else 0
    spk_reserve = 1 if spk_emb is not None else 0
    fill_end = start_pos
    fill_start = max(spk_reserve, start_pos - ref_len)
    if ref_codes is not None and fill_start < fill_end:
        audio_buffer[:, :, fill_start:fill_end] = ref_codes[:, :, -(fill_end - fill_start) :]
    if spk_emb is not None and fill_start > 0:
        audio_buffer[:, :, fill_start - 1] = audio_spk

    think_end_step, generated_tokens = None, ([] if args.get("open_thinking", False) else None)
    current_len = start_pos
    while current_len < capacity:
        if past_kvs is None or not use_cache:
            model_input = torch.cat(
                (audio_buffer[:, :, :current_len], text_buffer[:, :current_len].unsqueeze(1)),
                dim=1,
            )
        else:
            audio_input[:, :8, 0] = audio_buffer[:, :, current_len - 1]
            audio_input[:, 8, 0] = text_buffer[:, current_len - 1]
            model_input = audio_input
        out = model.forward(
            model_input,
            past_key_values=past_kvs,
            use_cache=use_cache,
            **args,
        )
        past_kvs = out.past_key_values

        logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
        # Keep repetition penalty on-device; .tolist() here forced a sync.
        logits[torch.unique(text_buffer[0, :current_len])] /= rp
        if top_p and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[1:], mask[0] = mask[:-1].clone(), False
            logits[sorted_indices[mask]] = -float("Inf")
        text_token = torch.multinomial(functional.softmax(logits, dim=-1), 1).item()

        if text_finished:
            text_token = (
                args.get("enter_token_id", 201) if first_finished else args.get("pad_token_id", 0)
            )
            first_finished = False

        step = current_len - start_pos
        audio_step = step - 1
        if generated_tokens is not None:
            generated_tokens.append(text_token)
            if not think_end_step and generated_tokens[-len(model.config.think_end_ids) :] == list(
                model.config.think_end_ids
            ):
                think_end_step = step + 2
            audio_step = (step - think_end_step) if think_end_step else -1

        active = [i for i in range(8) if audio_step >= i]
        sampled = _sample_audio_codes(out.audio_logits, audio_codes, active) if active else {}
        next_audio = [audio_pad] * 8
        for i in range(8):
            code = sampled.get(i, audio_pad)
            next_audio[i] = code
            audio_codes[i].append(code)
            if i in sampled and audio_stop_pos[i] is None and code >= 2048:
                audio_stop_pos[i] = len(audio_codes[i]) - 1

        if text_finished and audio_codes[7][-1] == audio_stop:
            break

        text_buffer[:, current_len] = text_token
        audio_buffer[:, :, current_len] = torch.tensor(
            next_audio, dtype=torch.long, device=input_ids.device
        )
        current_len += 1

        audio_frame = None
        if return_audio_codes and audio_step >= 7:
            frame = [audio_codes[i][step - 7 + i] for i in range(8)]
            active_layers = sum(
                1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i]
            )
            if active_layers >= 8:
                audio_frame = frame
        if not text_finished:
            yield text_buffer[:, start_pos:current_len], audio_frame
            if text_token == eos_token_id:
                text_finished = True
        else:
            yield None, audio_frame


__all__ = ["stream_generate_optimized"]
