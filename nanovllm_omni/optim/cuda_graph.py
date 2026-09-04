"""CUDA Graph streaming decode for MiniMind-O (plan §3.3).

Wraps the fixed-KV-buffer attention (``enable_fixed_kv_buffer``, report
§23-§25) in a per-step CUDA Graph capture/replay loop, collapsing the
~8 000 per-generate `cudaLaunchKernel` calls into a handful of graph
replays (~7.5x measured on RTX 3050, report §24).

Protocol (locked by GPU experiments in
`docs/perf/ncu-generate-kernels-2026-09-01.md`):

- **capture once, replay many** (§25): re-capturing per run folds
  capture-time RNG/state side effects into the loop and breaks
  determinism. A decoder captures `n_steps` per-step graphs the first
  time it runs, then replays them on every later call.
- host-side multinomial sampling lives OUTSIDE the graphs (§24/§25 keep
  sampling's RNG host-side and deterministic).
- the two freqs host-reads in the upstream forward are the only capture
  blockers (§13/§18). This module neutralizes them by compiling a copy of
  the forward with the checks set to ``if False`` (warmup proves them dead)
  and calling that copy — it never rebinds the model class.

Opt-in: ``enable_cuda_graph`` returns ``None`` when CUDA is unavailable or
the model isn't the MiniMind-O upstream shape. The caller keeps the eager
path in that case.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any

import torch

from nanovllm_omni.models.minimind_omni._sampling import (
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_P,
    sample_one_audio_layer,
    sample_text_token,
)
from nanovllm_omni.models.minimind_omni.attention import enable_fixed_kv_buffer

_log = logging.getLogger(__name__)

_ENABLE_MARKER = "_nanovllm_cuda_graph"


def _patched_forward(model: Any, cls: Any, src: str) -> Any:
    """Compile a copy of cls.forward with the two freqs `[0,0]` host-read
    checks neutralized (proven capture blockers, §13/§18). Returns a plain
    function `fwd(model, **kwargs)` — never rebinds the class."""
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    if "if False:" not in patched:
        _log.warning("enable_cuda_graph: no freqs host-reads to neutralize; skipping")
        return None
    namespace = dict(cls.forward.__globals__)
    namespace["__name__"] = cls.__module__
    namespace["__qualname__"] = cls.__qualname__ + ".forward_graph"
    exec(compile(textwrap.dedent(patched), "<enable_cuda_graph>", "exec"), namespace)
    return namespace["forward"]


@dataclass
class _GraphStep:
    graph: Any
    input: Any
    output: Any


class CudaGraphDecoder:
    """Capture-once / replay-many CUDA-Graphed decoder over the buffer-ized
    model's per-instance KV buffers."""

    def __init__(
        self,
        model: Any,
        fwd: Any,
        n_steps: int,
        *,
        temperature: float = DEFAULT_TEXT_TEMPERATURE,
        top_p: float = DEFAULT_TEXT_TOP_P,
        rp: float = 1.0,
    ) -> None:
        self.model = model
        self.fwd = fwd
        self.n_steps = n_steps
        self.temperature = temperature
        self.top_p = top_p
        self.rp = rp
        self.audio_pad = int(model.config.audio_pad_token)
        self.attns = [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]
        self.steps: list[_GraphStep] = []
        self._captured = False
        self._captured_len = -1
        self._prefill_len = -1
        self._last_input_ids = None

    # -- stateless helpers ------------------------------------------------
    @staticmethod
    def _decode_input(nid: torch.Tensor, audio_pad: int) -> torch.Tensor:
        buf = torch.full((1, 8, 1), audio_pad, dtype=torch.long, device=nid.device)
        return torch.cat((buf, nid.unsqueeze(-1)), dim=1)

    @staticmethod
    def _prefill_input(input_ids: torch.Tensor, audio_pad: int) -> torch.Tensor:
        """Production prefill input: [1, 9, seq] = 8 audio-pad rows + the text
        row, matching stream_generate's first forward (fact #3: the KV state
        must include the audio rows or decode logits diverge from index 1)."""
        buf = torch.full(
            (1, 8, input_ids.shape[1]), audio_pad, dtype=torch.long, device=input_ids.device
        )
        return torch.cat((buf, input_ids.unsqueeze(1)), dim=1)

    def _reset_buffers(self) -> None:
        for a in self.attns:
            a._kv_pos = 0

    def _clear_buffers(self) -> None:
        """Zero the full per-attention KV buffer. Each generate must start a
        fresh KV session (like a new production conversation): stale KV from
        a prior prompt must never survive into the current one. Graced with
        in-place zero; keeps buffer addresses stable for the captured graphs."""
        for a in self.attns:
            if hasattr(a, "_kv_past_key"):
                a._kv_past_key.zero_()
                a._kv_past_value.zero_()

    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run production-shape prefill and return the last text-row logits.
        Fact #3: prefill with the [1, 9, seq] input (audio pads + text) so
        the KV state matches stream_generate's; token0 is sampled from the
        text row's logits."""
        self._prefill_len = input_ids.shape[1]
        self._last_input_ids = input_ids
        self._reset_buffers()
        prefill_in = self._prefill_input(input_ids, self.audio_pad)
        with torch.no_grad():
            out = self.fwd(self.model, input_ids=prefill_in, past_key_values=None, use_cache=True)
        # text row is the last channel: logits[0, -1] is the last text position
        return out.logits[0, -1].clone()

    def _needs_recapture(self) -> bool:
        """Pure decision: must the per-step graphs be rebuilt?

        True when not yet captured OR the current prompt length differs from
        the length the existing graphs were captured at (defect #5: offsets
        are baked at that length; a different-length prompt must rebuild).
        Kept as its own method so the CPU contract test exercises the REAL
        code path (not a hand-replicated copy).
        """
        return not (self._captured and self._captured_len == self._prefill_len)

    def _capture(self, nid: torch.Tensor) -> None:
        """Capture one graph per AR step over the (prefill-loaded) KV buffers.

        Defect #5: re-capture when the prefill length changed since the last
        capture. Per-step graph offsets are baked at ``_kv_pos = prefill_len
        + k``; reusing them for a different-length prompt misaligns KV, so a
        length change must drop the stale graphs and rebuild.
        """
        if not self._needs_recapture():
            return
        # prompt length changed -> stale per-step graphs are unsafe. The
        # buffer still holds the PREVIOUS prompt's KV in [min_len : max_len]
        # where the new prefill didn't overwrite. Clear it here (re-capture
        # branch only, same-length reuse stays un-cleared) then re-prefill the
        # current prompt so the rebuilt graphs start from clean KV.
        self.steps = []
        self._captured = False
        self._clear_buffers()
        self._prefill(self._last_input_ids)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.n_steps):
                inp = self._decode_input(nid, self.audio_pad).clone()
                with torch.no_grad():
                    self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
                g = torch.cuda.CUDAGraph()
                holder: dict[str, Any] = {}
                with torch.cuda.graph(g), torch.no_grad():
                    holder["out"] = self.fwd(
                        self.model, input_ids=inp, past_key_values=None, use_cache=True
                    )
                self.steps.append(_GraphStep(graph=g, input=inp, output=holder["out"]))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self._captured = True
        self._captured_len = self._prefill_len

    # -- programmatic API -------------------------------------------------
    def generate_tokens(
        self, input_ids: torch.Tensor, *, seed: int | None = None, return_audio: bool = False
    ) -> list[int] | tuple[list[int], list[list[int]]]:
        """Prefill + n-step joint text/audio decode; returns generated text
        token ids (excludes input), and (if ``return_audio``) the 8-channel
        Mimi audio codes.

        Fact #4 mechanism: production draws TEXT then 8 AUDIO codes per step
        through ONE shared per-request ``torch.Generator`` (runner
        ``_sample_text``/``_sample_audio_row`` both pass ``gen=st.gen``).
        To keep the text sequence identical to ``stream_generate``, this
        decoder draws the audio codes through the same ``gen`` (advancing
        its state exactly like production) even when the caller only wants
        text. For WAV output the audio codes are returned as well.
        """
        num_layers = 8
        prefill_logits = self._prefill(input_ids)
        history = list(input_ids.reshape(-1).tolist())
        audio_history: list[list[int]] = [[] for _ in range(num_layers)]
        gen = torch.Generator(device=prefill_logits.device)
        gen.manual_seed(seed if seed is not None else torch.initial_seed())

        # seed0: sampled from prefill logits; audio_step -1 => all audio pad
        tok = sample_text_token(
            prefill_logits,
            history_ids=history,
            temperature=self.temperature,
            top_p=self.top_p,
            rp=self.rp,
            gen=gen,
        )
        text_codes = [tok]
        audio_codes = [[self.audio_pad] for _ in range(num_layers)]
        history = history + [tok]
        placeholder = torch.tensor([[tok]], device=prefill_logits.device, dtype=torch.long)
        self._capture(placeholder)
        cur = placeholder
        for k in range(self.n_steps - 1):
            step_index = k + 1  # text token index at this decode step
            audio_step = step_index - 1  # audio lags text by one position
            self.steps[k].input.copy_(self._decode_input(cur, self.audio_pad))
            self.steps[k].graph.replay()
            out = self.steps[k].output
            tok = sample_text_token(
                out.logits[0, -1],
                history_ids=history,
                temperature=self.temperature,
                top_p=self.top_p,
                rp=self.rp,
                gen=gen,
            )
            text_codes.append(tok)
            history = history + [tok]
            # audio draws (same gen => identical RNG advance to production)
            audio_logits = getattr(out, "audio_logits", None)
            for layer in range(num_layers):
                if audio_step >= layer and audio_logits is not None:
                    code = sample_one_audio_layer(
                        audio_logits[layer][0, -1],
                        audio_history[layer],
                        gen=gen,
                    )
                    audio_codes[layer].append(code)
                    audio_history[layer].append(code)
                else:
                    audio_codes[layer].append(self.audio_pad)
            cur = torch.tensor([[tok]], device=out.logits.device, dtype=torch.long)
        if return_audio:
            return text_codes, audio_codes
        return text_codes


def enable_cuda_graph(
    model: Any, n_steps: int = 16, max_len: int | None = None
) -> CudaGraphDecoder | None:
    """Install fixed-KV-buffer attention + return a CUDA-Graphed decoder.

    Returns ``None`` (not enabled) when CUDA is unavailable or the model
    doesn't carry the MiniMind-O freqs host-reads this integration targets.
    Opt-in: does not touch the bundle default path.
    """
    if not torch.cuda.is_available():
        _log.info("enable_cuda_graph: no CUDA; keeping eager path")
        return None
    if not hasattr(model.config, "audio_pad_token"):
        _log.warning("enable_cuda_graph: no audio_pad_token; skipping")
        return None
    if getattr(model, _ENABLE_MARKER, False):
        return model._nanovllm_graph_decoder

    cls = type(model)
    try:
        src = inspect.getsource(cls.forward)
    except (OSError, TypeError) as exc:
        _log.warning("enable_cuda_graph: cannot read forward source: %s", exc)
        return None
    fwd = _patched_forward(model, cls, src)
    if fwd is None:
        return None

    enable_fixed_kv_buffer(model, max_len)
    enabled = [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]
    if not enabled:
        _log.warning("enable_cuda_graph: no attention instances buffer-ized; skipping")
        return None
    _log.info("enable_cuda_graph: %d attention instances buffer-ized", len(enabled))

    decoder = CudaGraphDecoder(model=model, fwd=fwd, n_steps=n_steps)
    setattr(model, _ENABLE_MARKER, True)
    model._nanovllm_graph_decoder = decoder
    return decoder
