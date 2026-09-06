"""MiniMind-O talker stage.

Wraps the vendored HF ``TalkerModule`` (``model.talker``) into an
LLM_AR-shaped class for the 3-stage pipeline. Public symbols:
``MiniMindOmniTalkerForConditionalGeneration``, ``TalkerOutput``,
``wrap_talker``, ``_drive_talker_generation``, ``_talker_stage``.

Naming follows AGENTS.md (``num_*`` / ``sequence_*`` / ``token_*``);
vendored HF Attention's ``n_rep`` / ``n_local_heads`` are read-only
external-contract attributes and stay as-is.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output envelope (local replacement for vllm-omni's OmniOutput template)
# ---------------------------------------------------------------------------


@dataclass
class TalkerOutput:
    """Output envelope returned by :meth:`MiniMindOmniTalkerForConditionalGeneration.make_omni_output`.

    Mirrors the consumer-visible fields of vllm-omni's ``OmniOutput``:
    text-style hidden states plus multimodal (codec) outputs.
    """

    text_hidden_states: torch.Tensor | None = None
    multimodal_outputs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MiniMindOmniTalkerForConditionalGeneration(nn.Module):
    """Aligned talker wrapper. See module docstring for port scope.

    Construct from a loaded :class:`nanovllm_omni.models.minimind_omni.bundle.MinimindBundle`
    or directly from an HF ``TalkerModule``. Holds shared references to
    the inner module's ``layers`` / ``norm`` / ``lm_head`` / ``embed_tokens``
    / ``codec_proj`` / ``embed_proj`` / ``text_scale`` / ``audio_scale`` /
    ``spk_proj`` so checkpoint loading via :meth:`load_weights` writes
    into the right buffers and the fused-projection patch in
    ``attention.enable_fused_projections`` still takes effect.
    """

    # HF checkpoint prefix mapping: HF stores everything under
    # ``model.talker.*``; we strip ``model.talker.`` so wrapper params
    # keep their natural names (``layers.0.self_attn.q_proj.weight``).
    hf_to_local_prefix_map: dict[str, str] = {"model.talker.": ""}

    def __init__(
        self,
        bundle: Any | None = None,
        *,
        hf_talker: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hf_talker is None:
            if bundle is None or not hasattr(bundle, "model"):
                raise ValueError(
                    "MiniMindOmniTalkerForConditionalGeneration requires a "
                    "MinimindBundle or an explicit ``hf_talker`` argument."
                )
            hf_talker = getattr(bundle.model, "talker", None)
            if hf_talker is None:
                raise ValueError("bundle.model.talker is missing; HF TalkerModule not loaded.")
        # ``object.__setattr__`` bypasses ``nn.Module.__setattr__`` so the
        # inner module is reachable but invisible to the module tree walk
        # (avoids duplicate parameter paths in ``named_parameters()``).
        object.__setattr__(self, "_inner", hf_talker)
        self._bundle = bundle
        self._config = getattr(bundle.model, "config", None) if bundle is not None else None

        # Mirror config fields the talker needs at runtime.
        # Read from the OmniConfig (or a duck-typed object); defaults
        # match MiniMind-O's published values so a fake config in tests still
        # constructs without a real checkpoint.
        self.audio_pad_token: int = int(
            getattr(self._config, "audio_pad_token", 2049) if self._config else 2049
        )
        self.audio_stop_token: int = int(
            getattr(self._config, "audio_stop_token", 2050) if self._config else 2050
        )
        self.audio_spk_token: int = int(
            getattr(self._config, "audio_spk_token", 2051) if self._config else 2051
        )
        # Default ``internal_stop_token_id`` to ``audio_stop_token`` (vendored
        # HF OmniConfig does not expose one).
        self.internal_stop_token_id: int = (
            getattr(self._config, "internal_stop_token_id", self.audio_stop_token)
            if self._config
            else self.audio_stop_token
        )
        if isinstance(self.internal_stop_token_id, bool) or not isinstance(
            self.internal_stop_token_id, int
        ):
            raise ValueError(
                "internal_stop_token_id must be an integer, " f"got {self.internal_stop_token_id!r}"
            )
        self.audio_vocab_size: int = int(
            getattr(self._config, "audio_vocab_size", 2112) if self._config else 2112
        )
        self.num_code_layers: int = int(
            getattr(self._config, "num_code_layers", 8) if self._config else 8
        )
        # Watchdog length (vllm-omni publishes 192; vendored HF has no knob).
        self.max_steps_after_last_thinker_token: int = (
            getattr(self._config, "talker_max_steps_after_last_thinker_token", 192)
            if self._config
            else 192
        )
        if isinstance(self.max_steps_after_last_thinker_token, bool) or not isinstance(
            self.max_steps_after_last_thinker_token, int
        ):
            raise ValueError(
                "talker_max_steps_after_last_thinker_token must be an integer, "
                f"got {self.max_steps_after_last_thinker_token!r}"
            )
        self._steps_after_last_thinker_by_req: dict[str, int] = {}
        self.hidden_size: int = int(
            getattr(self._config, "talker_hidden_size", 768) if self._config else 768
        )
        self.text_hidden_size: int = int(
            getattr(self._config, "hidden_size", self.hidden_size)
            if self._config
            else self.hidden_size
        )
        self.spk_emb_size: int = int(
            getattr(self._config, "spk_emb_size", 192) if self._config else 192
        )
        self.max_position_embeddings: int = int(
            getattr(self._config, "max_position_embeddings", 32768) if self._config else 32768
        )
        self.rms_norm_eps: float = float(
            getattr(self._config, "rms_norm_eps", 1e-6) if self._config else 1e-6
        )
        self.use_moe: bool = bool(
            getattr(self._config, "use_moe", False) if self._config else False
        )

        # Re-expose inner-module attributes so callers reaching
        # ``talker.layers`` / ``talker.lm_head`` keep working without a
        # wrapper-aware rewrite. Inner module is the source of truth --
        # parameter fusion in ``attention.enable_fused_projections``
        # mutates these same attributes in place.
        self.layers = hf_talker.layers
        self.norm = hf_talker.norm
        self.lm_head = hf_talker.lm_head
        self.embed_tokens = hf_talker.embed_tokens
        self.codec_proj = hf_talker.codec_proj
        self.embed_proj = hf_talker.embed_proj
        self.text_scale = hf_talker.text_scale
        self.audio_scale = hf_talker.audio_scale
        self.spk_proj = hf_talker.spk_proj
        # RoPE buffers live on the inner module; share the references so any
        # buffer recompute on the inner module propagates here.
        self.freqs_cos = getattr(hf_talker, "freqs_cos", None)
        self.freqs_sin = getattr(hf_talker, "freqs_sin", None)

        # LLM_AR bookkeeping state.
        self._stop_pending_by_req: dict[str, bool] = {}
        self._build_code_layer_masks()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_code_layer_masks(self) -> None:
        """Precompute the [num_code_layers+1, num_code_layers] active mask.

        ``step_ids = [-1, 0, 1, ..., num_code_layers-1]`` vs
        ``layer_ids = [0, 1, ..., num_code_layers-1]``; entry ``step_id``
        is True for layers ``<= step_id``.
        """
        layer_ids = torch.arange(self.num_code_layers)
        step_ids = torch.arange(self.num_code_layers + 1) - 1
        # persistent=False: the mask is recomputable from config.
        self.register_buffer(
            "_code_layer_masks",
            (step_ids.unsqueeze(-1) >= layer_ids.unsqueeze(0)),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Input / output helpers (port from vllm-omni talker)
    # ------------------------------------------------------------------

    def _audio_ids_from_layer0(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Split ``[B, T]`` text-style input ids into ``[B, num_code_layers, T]``.

        Layer 0 carries the input ids (clamped into ``[0, audio_vocab_size)``);
        the remaining layers are filled with the audio pad token. A 3-D input
        (already-split audio rows) is passed through so callers that build
        the rectangular input themselves can reuse this helper.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim == 3:
            return input_ids.to(dtype=torch.long)
        batch_size, sequence_len = input_ids.shape
        audio_ids = torch.full(
            (batch_size, self.num_code_layers, sequence_len),
            self.audio_pad_token,
            dtype=torch.long,
            device=input_ids.device,
        )
        audio_ids[:, 0, :] = input_ids.to(dtype=torch.long).clamp(
            min=0, max=self.audio_vocab_size - 1
        )
        return audio_ids

    def _select_bridge_states(
        self,
        info_dict: dict[str, Any],
        span_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, bool, int, int, int]:
        """Pick the bridge hidden-state slice for this forward call.

        Returns ``(bridge_span, is_prefill, prompt_len, num_computed, bridge_len)``.
        The span is sliced so that decode inputs at position ``num_computed``
        condition on the matching bridge row, not always ``bridge[-1]`` --
        the same alignment vllm-omni enforces.
        """
        hidden_info = info_dict.get("hidden_states", {}) if isinstance(info_dict, dict) else {}
        bridge = hidden_info.get("bridge") if isinstance(hidden_info, dict) else None
        if not isinstance(bridge, torch.Tensor):
            raise ValueError(
                "MiniMind talker requires hidden_states.bridge tensor in additional_information."
            )
        # Coerce dtype/device to the talker's parameter dtype.
        ref_param = next(self.parameters())
        target_dtype = ref_param.dtype
        if bridge.device != device or bridge.dtype != target_dtype:
            bridge = bridge.to(device=device, dtype=target_dtype, non_blocking=True)
            hidden_info["bridge"] = bridge
        if bridge.ndim == 3:
            bridge = bridge.reshape(-1, bridge.shape[-1])
        bridge_len = int(bridge.shape[0])
        if bridge.shape[0] < span_len:
            raise ValueError(
                f"bridge hidden states length {bridge.shape[0]} is shorter "
                f"than scheduled span {span_len}"
            )

        prompt_len = int(info_dict.get("_omni_prompt_len", 0) or 0)
        ids = info_dict.get("ids") if isinstance(info_dict, dict) else None
        if isinstance(ids, dict):
            text_prompt_len = len(ids.get("prompt") or [])
            if text_prompt_len > 0:
                prompt_len = text_prompt_len

        raw_num_computed = info_dict.get("_omni_num_computed_tokens")
        num_computed = prompt_len if raw_num_computed is None else int(raw_num_computed)
        raw_is_prefill = info_dict.get("_omni_is_prefill")
        is_prefill = num_computed < prompt_len if raw_is_prefill is None else bool(raw_is_prefill)

        start = max(0, min(num_computed, max(0, bridge.shape[0] - span_len)))
        end = start + span_len
        return bridge[start:end], is_prefill, prompt_len, num_computed, bridge_len

    def _make_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        bridge_states: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine audio-token + bridge hidden state into the talker's input embeds.

        Mirrors vllm-omni: ``embed_proj(bridge) * text_scale + codec_proj(audio_emb) * audio_scale``.
        ``spk_emb`` is projected into the spk-token position when supplied
        (the talker is conditioned on a single speaker-embedding token at
        the start of prefill).
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        audio_ids = self._audio_ids_from_layer0(input_ids)
        talker_emb = self.embed_tokens(audio_ids)
        if spk_emb is not None:
            if spk_emb.ndim == 1:
                spk_emb = spk_emb.unsqueeze(0)
            spk_emb = spk_emb.to(device=talker_emb.device, dtype=talker_emb.dtype)
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        text_part = self.embed_proj(
            bridge_states.to(device=talker_emb.device, dtype=talker_emb.dtype)
        )
        codec_part = self.codec_proj(talker_emb.reshape(-1, talker_emb.shape[-1]))
        return text_part * self.text_scale + codec_part * self.audio_scale

    def _sample_codebook_logits_batch(
        self,
        logits_by_layer: list[torch.Tensor],
        *,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_k: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one token per codebook layer per request row.

        Output shape: ``[num_layers, batch]`` (transposed so the caller's
        ``audio_codes`` matches MiniMind's [T, 8] convention -- one column
        per codebook layer, one row per decode step). Layer 0 is sampled
        by the engine's ``sample`` method, not here; this helper covers
        the residual layers for the full-code / MTP integration.
        """
        if not logits_by_layer:
            return torch.empty((0, 0), dtype=torch.long)
        logits = torch.stack(logits_by_layer, dim=0)
        num_layers, batch, vocab = logits.shape
        flat = logits.reshape(num_layers * batch, vocab).float()
        if 0 <= self.audio_pad_token < vocab:
            flat[:, self.audio_pad_token] = -float("inf")
        if not do_sample:
            sampled = flat.argmax(dim=-1)
        else:
            temperature = max(float(temperature), 1e-5)
            flat = flat / temperature
            if 0 < top_k < vocab:
                top_val, top_idx = flat.topk(top_k, dim=-1)
                sample = torch.multinomial(torch.softmax(top_val, dim=-1), 1, generator=generator)
                sampled = top_idx.gather(-1, sample).squeeze(-1)
            else:
                sampled = torch.multinomial(
                    torch.softmax(flat, dim=-1), 1, generator=generator
                ).squeeze(-1)
        return sampled.reshape(num_layers, batch).transpose(0, 1).contiguous()

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
        temperature: float = 0.2,
        top_k: int = 50,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate one delayed diagonal row of audio codebooks.

        ``input_ids`` supplies the current layer-0 code and the adapter heads
        predict layers ``1..num_code_layers-1`` from the previous talker
        hidden state. ``active_mask`` is exact ``[B, num_code_layers]`` for
        batched calls; a one-dimensional mask is accepted only for ``B=1``.
        """
        if input_ids.ndim == 1:
            if input_ids.numel() == 0:
                raise ValueError("input_ids must contain at least one row")
            batch_size = int(input_ids.shape[0])
            layer0 = input_ids
        elif input_ids.ndim == 2 and input_ids.shape[1] == 1:
            batch_size = int(input_ids.shape[0])
            layer0 = input_ids[:, 0]
        else:
            raise ValueError("input_ids must have shape [batch_size] or [batch_size, 1]")

        if input_embeds.ndim < 1 or input_embeds.shape[0] != batch_size:
            raise ValueError(
                f"input_embeds must have first dimension {batch_size}, "
                f"got {tuple(input_embeds.shape)}"
            )
        if last_talker_hidden.ndim == 1:
            if batch_size != 1:
                raise ValueError("last_talker_hidden must have one row per input_ids row")
        elif last_talker_hidden.shape[0] != batch_size:
            raise ValueError("last_talker_hidden must have one row per input_ids row")
        if text_step.ndim == 1:
            if batch_size != 1:
                raise ValueError("text_step must have one row per input_ids row")
        elif text_step.shape[0] != batch_size:
            raise ValueError("text_step must have one row per input_ids row")

        output_device = input_embeds.device
        hidden = last_talker_hidden.reshape(batch_size, -1).to(
            device=output_device, dtype=input_embeds.dtype
        )
        logits_by_layer = self.lm_head(hidden)
        if len(logits_by_layer) != self.num_code_layers:
            raise ValueError(
                f"lm_head returned {len(logits_by_layer)} code layers; "
                f"expected {self.num_code_layers}"
            )
        residual_codes = self._sample_codebook_logits_batch(
            logits_by_layer[1:],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        audio_codes = torch.cat(
            (
                layer0.to(device=output_device, dtype=torch.long).reshape(batch_size, 1),
                residual_codes.to(device=output_device, dtype=torch.long),
            ),
            dim=1,
        )

        if active_mask is not None:
            if active_mask.ndim == 1:
                if batch_size != 1 or active_mask.shape[0] != self.num_code_layers:
                    raise ValueError(
                        "active_mask must have shape " f"[{batch_size}, {self.num_code_layers}]"
                    )
                active_mask = active_mask.unsqueeze(0)
            elif active_mask.ndim != 2 or active_mask.shape != (
                batch_size,
                self.num_code_layers,
            ):
                raise ValueError(
                    "active_mask must have shape "
                    f"[{batch_size}, {self.num_code_layers}], got {tuple(active_mask.shape)}"
                )
            active = active_mask.to(device=output_device, dtype=torch.bool)
            audio_codes = audio_codes.masked_fill(~active, self.audio_pad_token)
        return audio_codes

    # ------------------------------------------------------------------
    # LLM_AR interface (mirrors vllm-omni's stage contract)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Pre-thinker-vllm helper: text-style embed for codec tokens.

        Mirrors vllm-omni's ``embed_input_ids`` shape: returns the
        ``[span, hidden_size]`` audio-side embedding.
        """
        audio_ids = self._audio_ids_from_layer0(input_ids)
        return self.codec_proj(self.embed_tokens(audio_ids)).reshape(-1, self.hidden_size)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        _input_embeds: torch.Tensor | None = None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Pre-forward bridge: pull bridge states from ``info_dict``, build embeds.

        Returns ``(input_ids, embeds, update)`` where ``update`` carries
        prefill padding + MTP inputs.
        """
        span_len = int(input_ids.shape[0])
        bridge_states, is_prefill, prompt_len, num_computed, bridge_len = (
            self._select_bridge_states(info_dict, span_len, input_ids.device)
        )
        spk_emb = info_dict.get("spk_emb")
        embeds = self._make_inputs_embeds(
            input_ids.view(1, -1),
            bridge_states,
            spk_emb=spk_emb,
        )
        update: dict[str, Any] = {}
        if span_len == 1:
            hidden = info_dict.get("hidden_states", {})
            last_hidden = hidden.get("last") if isinstance(hidden, dict) else None
            if not isinstance(last_hidden, torch.Tensor):
                last_hidden = torch.zeros(
                    (1, self.hidden_size),
                    device=input_ids.device,
                    dtype=embeds.dtype,
                )
            text_step = self.embed_proj(bridge_states.to(device=embeds.device, dtype=embeds.dtype))
            audio_step = max(0, num_computed - prompt_len) - 1
            mask_idx = max(0, min(audio_step + 1, self.num_code_layers))
            active_mask = self._code_layer_masks[mask_idx : mask_idx + 1].to(device=embeds.device)
            update["mtp_inputs"] = (
                last_hidden.reshape(1, -1).to(device=embeds.device, dtype=embeds.dtype),
                text_step.reshape(1, -1),
                active_mask,
            )
            if not is_prefill:
                request_id = info_dict.get("request_id")
                if isinstance(request_id, str) and num_computed >= bridge_len:
                    steps_after_last_thinker = (
                        self._steps_after_last_thinker_by_req.get(request_id, 0) + 1
                    )
                    self._steps_after_last_thinker_by_req[request_id] = steps_after_last_thinker
                    if (
                        self.max_steps_after_last_thinker_token >= 0
                        and steps_after_last_thinker >= self.max_steps_after_last_thinker_token
                    ):
                        self._stop_pending_by_req[request_id] = True
        else:
            update.setdefault("codes", {})["audio"] = torch.full(
                (span_len, self.num_code_layers),
                self.audio_pad_token,
                dtype=torch.long,
                device=input_ids.device,
            )
        return input_ids, embeds, update

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Run the talker trunk and return the final hidden state.

        ``positions`` is accepted for API parity. The vendored HF
        :class:`MiniMindBlock` takes ``(hidden_states, position_embeddings)``
        where ``position_embeddings`` is ``(cos, sin)``; we pass the
        freqs_cos/sin buffers sliced by the current ``positions`` (or all
        of them when positions is None).
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embed_input_ids(input_ids)
        # The vendored HF ``MiniMindBlock`` expects ``[B, T, H]``;
        # ``preprocess`` hands us 2D ``[T, H]``. Add the batch dim for the
        # trunk and drop it again on return so the public 2D->2D contract
        # is preserved (3D in -> 3D out).
        had_batch_dim = inputs_embeds.ndim >= 3
        hidden_states = inputs_embeds if had_batch_dim else inputs_embeds.unsqueeze(0)
        # Determine start_pos for RoPE slicing.
        if positions is None:
            start_pos = 0
            span_len = int(hidden_states.shape[1]) if hidden_states.ndim >= 2 else 1
        else:
            positions_t = (
                positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
            )
            start_pos = int(positions_t.min().item()) if positions_t.numel() > 0 else 0
            span_len = (
                int(positions_t.max().item()) - start_pos + 1 if positions_t.numel() > 0 else 1
            )
        position_embeddings = self._slice_rope(start_pos, span_len, hidden_states.device)
        for layer in self.layers:
            if position_embeddings is not None:
                hidden_states, _ = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=None,
                    use_cache=False,
                )
            else:
                hidden_states, _ = layer(
                    hidden_states,
                    None,
                    past_key_value=None,
                    use_cache=False,
                )
        out = self.norm(hidden_states)
        return out if had_batch_dim else out.squeeze(0)

    def _slice_rope(
        self, start_pos: int, span_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Slice ``freqs_cos`` / ``freqs_sin`` for the current decode step."""
        cos = self.freqs_cos
        sin = self.freqs_sin
        if cos is None or sin is None:
            return None
        cos = cos[start_pos : start_pos + span_len].to(device=device, dtype=cos.dtype)
        sin = sin[start_pos : start_pos + span_len].to(device=device, dtype=sin.dtype)
        return cos, sin

    def compute_logits(self, hidden_states: torch.Tensor | TalkerOutput) -> torch.Tensor | None:
        """Project the talker hidden state to layer-0 audio logits.

        Residual adapter heads are consumed by ``talker_mtp`` for delayed
        full-code output.
        """
        if isinstance(hidden_states, TalkerOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.lm_head(hidden_states)[0]

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Sample one audio token per request row from layer-0 logits.

        Uses simple multinomial sampling with the metadata's ``temperature`` /
        ``top_k`` / ``generator`` (when supplied). ``internal_stop_token_id``
        is force-masked to ``-inf``.
        """
        if logits is None:
            return None
        # Build per-row sampling kwargs from the metadata duck type.
        temperature = float(getattr(sampling_metadata, "temperature", 1.0) or 1.0)
        top_k = int(getattr(sampling_metadata, "top_k", 0) or 0)
        do_sample = bool(getattr(sampling_metadata, "do_sample", True))
        gen = getattr(sampling_metadata, "generator", None)
        request_ids = list(getattr(sampling_metadata, "request_ids", None) or [])

        logits_for_sample = logits
        if 0 <= self.internal_stop_token_id < logits.shape[-1]:
            logits_for_sample = logits.clone()
            logits_for_sample[:, self.internal_stop_token_id] = float("-inf")
        flat = logits_for_sample.float()
        if not do_sample:
            sampled = flat.argmax(dim=-1)
        else:
            t = max(temperature, 1e-5)
            flat = flat / t
            if top_k > 0 and top_k < flat.shape[-1]:
                top_v, top_i = flat.topk(top_k, dim=-1)
                sample = torch.multinomial(torch.softmax(top_v, dim=-1), 1, generator=gen)
                sampled = top_i.gather(-1, sample).squeeze(-1)
            else:
                sampled = torch.multinomial(torch.softmax(flat, dim=-1), 1, generator=gen).squeeze(
                    -1
                )
        sampled = sampled.reshape(-1, 1)

        # Apply forced internal-stop rows.
        for row in range(sampled.shape[0]):
            request_id = request_ids[row] if row < len(request_ids) else None
            if request_id is None or not self._stop_pending_by_req.get(request_id):
                continue
            sampled[row, 0] = self.internal_stop_token_id
            self._stop_pending_by_req.pop(request_id, None)
        return sampled

    # ------------------------------------------------------------------
    # Frame alignment (delayed diagonal MTP)
    # ------------------------------------------------------------------

    def _normalise_audio_code_rows(
        self,
        audio: Any,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """Coerce ``audio`` to ``[num_rows, num_code_layers]`` long tensor.

        Returns ``None`` for empty / wrong-rank inputs.
        """
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            return None
        rows = audio.to(device=device if device is not None else audio.device, dtype=torch.long)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[-1] != self.num_code_layers:
            return None
        return rows

    def _ready_diagonal_audio_frames(
        self,
        history: torch.Tensor | None,
        current: torch.Tensor,
        emitted_frames: int,
    ) -> torch.Tensor | None:
        """Extract the diagonal MTP-ready frames from the concatenated code rows.

        Skips frames whose last layer hit the stop boundary
        (``code >= audio_vocab_size``).
        """
        if history is not None:
            history = self._normalise_audio_code_rows(history, device=current.device)
        rows = current if history is None else torch.cat((history, current), dim=0)
        total_rows = int(rows.shape[0])
        ready_frames = max(0, total_rows - self.num_code_layers + 1)
        if ready_frames <= emitted_frames:
            return None
        frames: list[torch.Tensor] = []
        delay = self.num_code_layers - 1
        for frame_idx in range(emitted_frames, ready_frames):
            end = frame_idx + delay
            frame = torch.stack(
                [rows[end - delay + layer, layer] for layer in range(self.num_code_layers)]
            )
            # Every diagonal entry must be a sampled code, not a boundary row.
            if (
                (frame >= self.audio_vocab_size).any()
                or (frame == self.audio_pad_token).any()
                or (frame == self.audio_stop_token).any()
            ):
                continue
            frames.append(frame)
        if not frames:
            return None
        return torch.stack(frames, dim=0).to(dtype=torch.long)

    # ------------------------------------------------------------------
    # Post-forward / lifecycle / output / checkpoint loading
    # ------------------------------------------------------------------

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Post-forward: stash last hidden state, detect audio_stop, build code history.

        Returns an ``info_dict``-shaped update. ``hidden_states['last']``
        is the per-row last-position hidden state (consumed by the next
        call's ``mtp_inputs``). ``codes['history']`` is the cumulative
        ``[num_rows, num_code_layers]`` table consumed by code2wav.
        """
        if hidden_states.numel() == 0:
            return {}

        request_id = kwargs.get("request_id")
        is_prefill = bool(kwargs.get("_omni_is_prefill", False))
        update: dict[str, Any] = {"hidden_states": {"last": hidden_states[-1:].detach()}}
        if is_prefill:
            return update

        codes = kwargs.get("codes", {}) if isinstance(kwargs, dict) else {}
        audio_raw = codes.get("audio") if isinstance(codes, dict) else None
        current = self._normalise_audio_code_rows(audio_raw)
        if current is None:
            return update

        history_raw = codes.get("history") if isinstance(codes, dict) else None
        history = self._normalise_audio_code_rows(history_raw, device=current.device)
        rows = current if history is None else torch.cat((history, current), dim=0)
        if isinstance(request_id, str) and current[-1, -1].item() == self.audio_stop_token:
            self._stop_pending_by_req[request_id] = True
        ready_frames = max(0, int(rows.shape[0]) - self.num_code_layers + 1)
        update.setdefault("codes", {})["history"] = rows.detach()
        update.setdefault("meta", {})["emitted_audio_frames"] = ready_frames
        return update

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Drop per-request state on request completion.

        Called by the engine when a request leaves the running set (EOS,
        max-tokens, or explicit abort).
        """
        for req_id in finished_req_ids:
            self._stop_pending_by_req.pop(req_id, None)
            self._steps_after_last_thinker_by_req.pop(req_id, None)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | TalkerOutput,
        **_kwargs: Any,
    ) -> TalkerOutput:
        """Wrap ``model_outputs`` in the public :class:`TalkerOutput` envelope.

        Accepts either a raw hidden-state tensor or an already-built
        ``TalkerOutput`` (passthrough). When ``model_intermediate_buffer``
        is supplied in kwargs, ready diagonal frames are extracted and
        attached to ``multimodal_outputs['codes']['audio']``.
        """
        if isinstance(model_outputs, TalkerOutput):
            return model_outputs
        info_dicts = _kwargs.get("model_intermediate_buffer") or _kwargs.get(
            "runtime_additional_information"
        )
        audio_codes_list: list[torch.Tensor] = []
        if isinstance(info_dicts, list):
            for info in info_dicts:
                if not isinstance(info, dict):
                    continue
                codes = info.get("codes", {})
                if not isinstance(codes, dict):
                    continue
                audio_raw = codes.get("audio")
                history_raw = codes.get("history")
                current = self._normalise_audio_code_rows(audio_raw)
                if current is None:
                    continue
                meta = info.get("meta", {})
                emitted = int(meta.get("emitted_audio_frames", 0)) if isinstance(meta, dict) else 0
                frames = self._ready_diagonal_audio_frames(history_raw, current, emitted)
                if frames is not None and frames.numel() > 0:
                    audio_codes_list.append(frames)
        if not audio_codes_list:
            return TalkerOutput(text_hidden_states=model_outputs, multimodal_outputs={})
        audio_codes = torch.cat(audio_codes_list, dim=0)
        return TalkerOutput(
            text_hidden_states=model_outputs[: audio_codes.shape[0]],
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    def load_weights(self, weights: Any) -> set[str]:
        """Load HF checkpoint tensors into our wrapper.

        Accepts any iterable of ``(name, tensor)`` pairs (HF state-dict or
        any iterable with the same shape). Strips the ``model.talker.``
        prefix and the prefix-mapped equivalents; skips keys that belong
        to the thinker / audio_proj / vision_proj / rotary buffers.

        Returns the set of wrapper-local parameter names that were loaded.
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_weights: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = raw_name
            for old, _new in self.hf_to_local_prefix_map.items():
                if name.startswith(old):
                    name = name[len(old) :]
                    break
            # Skip keys we don't own.
            if name.startswith(
                (
                    "thinker.",
                    "audio_proj.",
                    "vision_proj.",
                    "rotary_emb.inv_freq",
                    "model.",
                )
            ):
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            try:
                param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
                loaded_weights.add(name)
            except Exception as exc:  # noqa: BLE001 -- narrow message only
                logger.warning(
                    "talker.load_weights: failed to copy %s (%s): %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        return loaded_weights


# ---------------------------------------------------------------------------
# Bundle wiring
# ---------------------------------------------------------------------------


def wrap_talker(bundle: Any) -> MiniMindOmniTalkerForConditionalGeneration:
    """Wrap ``bundle.model.talker`` (the HF ``TalkerModule``) into our class.

    Idempotent: if ``bundle.talker`` is already our wrapper, returns it.
    Also assigns ``bundle.talker`` so callers that read the attribute
    (instead of catching the return value) still see the wrapper.

    Callers that want the raw HF module should reach for
    ``bundle.model.talker`` instead.
    """
    existing = getattr(bundle, "talker", None)
    if isinstance(existing, MiniMindOmniTalkerForConditionalGeneration):
        return existing
    wrapped = MiniMindOmniTalkerForConditionalGeneration(bundle)
    with contextlib.suppress(Exception):
        bundle.talker = wrapped
    return wrapped


# ---------------------------------------------------------------------------
# Pipeline factory / process_input shims (glue layer preserved)
# ---------------------------------------------------------------------------


def _drive_talker_generation(
    talker: MiniMindOmniTalkerForConditionalGeneration,
    payload: Any,
    *,
    temperature: float = 0.2,
    top_k: int = 50,
    do_sample: bool = True,
    mtp_runner: Any = None,
) -> Any:
    """Run the talker wrapper over a ``TalkerInputPayload``; return code rows.

    Drives the LLM_AR contract the wrapper exposes (``preprocess`` /
    ``forward`` / ``postprocess`` / ``compute_logits`` / ``sample`` /
    ``talker_mtp``) with the reference delayed-diagonal MTP alignment:
    prefill over the bridge prompt span, then one decode step per output
    bridge row.

    ``mtp_runner`` defaults to ``talker`` (eager ``talker.talker_mtp``); when
    a ``TalkerMtpCudaGraph`` is supplied, the per-step MTP call is routed
    through ``graph.decode(...)`` (graph forces ``do_sample=False``).

    Returns a frame-major ``[frames, num_code_layers]`` long tensor --
    the Code2Wav stage's input contract.
    """
    from types import SimpleNamespace

    import torch

    request_id = payload.request_id or "mmo-full-talker"
    # Run on the talker's own parameter device, never the bridge's (a CPU
    # bridge must not force the whole talker chain onto CPU against CUDA
    # weights). Move the bridge to that device explicitly.
    device = next(talker.parameters()).device
    talker_dtype = next(talker.parameters()).dtype
    mtp_runner = talker if mtp_runner is None else mtp_runner
    prompt_ids = list(payload.prompt_token_ids)
    output_ids = list(payload.output_token_ids)
    all_ids = list(payload.text_token_ids)
    if not all_ids:
        all_ids = prompt_ids + output_ids
    prompt_len = len(prompt_ids)
    bridge = payload.bridge_states
    if bridge.ndim == 3 and bridge.shape[0] == 1:
        bridge = bridge[0]
    bridge = bridge.to(device=device, dtype=talker_dtype)
    num_decode_steps = max(0, int(bridge.shape[0]) - prompt_len)
    if num_decode_steps == 0:
        # Still emit one audio row so the stage hands code2wav a non-empty frame table.
        num_decode_steps = 1

    info: dict[str, Any] = {
        "hidden_states": {"bridge": bridge},
        "ids": {"prompt": prompt_ids, "output": output_ids, "all": all_ids},
        "request_id": request_id,
        "_omni_prompt_len": prompt_len,
    }
    if payload.speaker_embedding is not None:
        info["spk_emb"] = payload.speaker_embedding

    rows: list[torch.Tensor] = []
    try:
        # Prefill: run the whole prompt span through the trunk once and
        # keep the last hidden row as the seed for the first MTP step.
        prefill_ids = torch.full(
            (prompt_len,), talker.audio_pad_token, dtype=torch.long, device=device
        )
        info["_omni_num_computed_tokens"] = 0
        info["_omni_is_prefill"] = True
        _input_ids, embeds, _update = talker.preprocess(prefill_ids, None, **info)
        hidden = talker.forward(
            input_ids=None,
            positions=torch.arange(prompt_len, dtype=torch.long, device=device),
            inputs_embeds=embeds,
        )
        post = talker.postprocess(hidden, request_id=request_id, _omni_is_prefill=True)
        last_hidden = post.get("hidden_states", {}).get("last")
        if not isinstance(last_hidden, torch.Tensor):
            last_hidden = hidden[-1:]

        # Decode: one step per output bridge row; residuals from the previous
        # step's hidden state (delayed diagonal MTP).
        for step in range(num_decode_steps):
            num_computed = prompt_len + step
            prev_code = int(rows[-1][0]) if rows else talker.audio_pad_token
            step_ids = torch.tensor([prev_code], dtype=torch.long, device=device)
            info["hidden_states"] = {"bridge": bridge, "last": last_hidden}
            info["_omni_num_computed_tokens"] = num_computed
            info["_omni_is_prefill"] = False
            _step_ids, embeds, update = talker.preprocess(step_ids, None, **info)
            hidden = talker.forward(
                input_ids=None,
                positions=torch.tensor([num_computed], dtype=torch.long, device=device),
                inputs_embeds=embeds,
            )
            logits = talker.compute_logits(hidden)
            meta = SimpleNamespace(
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                generator=None,
                request_ids=[request_id],
            )
            layer0 = talker.sample(logits, meta)
            mtp_inputs = update.get("mtp_inputs")
            if isinstance(mtp_inputs, (tuple, list)) and len(mtp_inputs) >= 3:
                last_hidden_for_mtp, text_step, active_mask = mtp_inputs[:3]
            else:
                last_hidden_for_mtp, text_step, active_mask = last_hidden, None, None
            mtp_decode = getattr(mtp_runner, "decode", None)
            if mtp_decode is not None:
                row = mtp_decode(
                    input_ids=layer0,
                    input_embeds=embeds,
                    last_talker_hidden=last_hidden_for_mtp,
                    text_step=text_step,
                    active_mask=active_mask,
                    do_sample=do_sample,
                )
            else:
                row = mtp_runner.talker_mtp(
                    input_ids=layer0,
                    input_embeds=embeds,
                    last_talker_hidden=last_hidden_for_mtp,
                    text_step=text_step,
                    active_mask=active_mask,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                )
            rows.append(row[0])
            last_hidden = hidden[-1:]
            if int(row[0, -1].item()) == talker.audio_stop_token:
                break
    finally:
        # Drop per-request flags so a recycled request id never inherits
        # a stale forced-stop.
        talker.on_requests_finished([request_id])

    if not rows:
        return torch.empty((0, talker.num_code_layers), dtype=torch.long, device=device)
    return torch.stack(rows, dim=0)


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: real talker over ``TalkerInputPayload``.

    Consumes a ``TalkerInputPayload`` (produced by ``thinker2talker``),
    drives the talker wrapper + ``talker_mtp`` over the bridge hidden
    states, and returns a ``TalkerOutput`` whose
    ``multimodal_outputs['codes']['audio']`` carries ``[F, num_code_layers]``
    code rows for the code2wav stage.
    """
    extra = dict(getattr(args, "extra", None) or {})
    injected_talker = extra.get("talker")
    injected_bundle = extra.get("bundle")
    stage_cache: dict[str, Any] = {}
    use_talker_graph = bool(getattr(deploy, "use_talker_cuda_graph", False))

    def _resolve_talker() -> Any:
        if "talker" in stage_cache:
            return stage_cache["talker"]
        if injected_talker is not None:
            talker = injected_talker
        elif injected_bundle is not None:
            talker = getattr(injected_bundle, "talker", None)
            if talker is None:
                talker = wrap_talker(injected_bundle)
        else:
            from .bundle import create_bundle

            bundle = create_bundle(
                model_id=args.model,
                device=args.device,
                trust_remote_code=getattr(args, "trust_remote_code", True),
                dtype=getattr(args, "dtype", None),
            )
            talker = wrap_talker(bundle)
        stage_cache["talker"] = talker
        return talker

    def _resolve_mtp_runner() -> tuple[Any, bool]:
        """Return ``(runner, graph_engaged)``.

        When ``use_talker_graph`` is true and CUDA is available, ``runner``
        is a ``TalkerMtpCudaGraph`` wrapping ``talker.talker_mtp``; otherwise
        it is the raw talker and ``graph_engaged`` is false. Cached per-stage.
        """
        if "mtp_runner" in stage_cache:
            return stage_cache["mtp_runner"]
        talker = _resolve_talker()
        runner = talker
        if use_talker_graph:
            from nanovllm_omni.optim.talker_cuda_graph import (
                enable_talker_mtp_cuda_graph,
            )

            graph_runner = enable_talker_mtp_cuda_graph(
                talker,
                batch_sizes=(1,),
                temperature=0.2,
                top_k=50,
            )
            if graph_runner is not None:
                runner = graph_runner
        engaged = runner is not talker
        cached = (runner, engaged)
        stage_cache["mtp_runner"] = cached
        return cached

    def talker_forward_full(payload: Any, sampling: Any) -> Any:
        from .stage_processors import TalkerInputPayload

        if not isinstance(payload, TalkerInputPayload):
            raise TypeError(
                "MiniMind full-mode talker stage requires a TalkerInputPayload "
                f"(thinker2talker output); got {type(payload).__name__}."
            )
        temperature = (
            float(getattr(sampling, "temperature", 0.2) or 0.2) if sampling is not None else 0.2
        )
        top_k = int((sampling.extra or {}).get("top_k", 50)) if sampling is not None else 50
        # ``do_sample`` lives in ``sampling.extra`` (deploy YAML default), so
        # a yaml can opt the talker stage into greedy. Defaults to True.
        do_sample_user = True
        if sampling is not None:
            if hasattr(sampling, "do_sample"):
                do_sample_user = bool(sampling.do_sample)
            elif (sampling.extra or {}).get("do_sample") is not None:
                do_sample_user = bool(sampling.extra["do_sample"])
        mtp_runner, graph_engaged = _resolve_mtp_runner()
        # CUDA Graphs can't capture stochastic multinomial, so the graph only
        # fires on greedy (do_sample=False). When the graph is engaged we force
        # greedy and log a one-time notice so a sampling request isn't silently
        # downgraded. Set use_talker_cuda_graph: false in the YAML to keep
        # sampling.
        if do_sample_user and graph_engaged:
            logger.info(
                "talker CUDA Graph engaged forces greedy decode; request "
                "do_sample=True ignored. Set use_talker_cuda_graph: false to "
                "keep sampling."
            )
        do_sample = do_sample_user and not graph_engaged
        codes = _drive_talker_generation(
            _resolve_talker(),
            payload,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            mtp_runner=mtp_runner,
        )
        if codes.shape[0] == 0:
            return TalkerOutput(text_hidden_states=None, multimodal_outputs={})
        return TalkerOutput(
            text_hidden_states=payload.bridge_states[: codes.shape[0]],
            multimodal_outputs={"codes": {"audio": codes}},
        )

    return talker_forward_full


def _identity_process_input(payload: Any, prompt: str) -> Any:
    """Default ``process_input``: pass the previous stage's output through unchanged.

    The MiniMind pipeline binds its real stage processors
    (``thinker2talker`` / ``talker2code2wav``) instead.
    """
    del prompt
    return payload


__all__ = [
    "MiniMindOmniTalkerForConditionalGeneration",
    "TalkerOutput",
    "wrap_talker",
    "_drive_talker_generation",
    "_identity_process_input",
    "_talker_stage",
]
