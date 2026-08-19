from __future__ import annotations

from ..engine_args import OmniEngineArgs


class OmniBase:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)
        self._bundle = None

    def _ensure_bundle(self):
        if self._bundle is None:
            from ..models.minimind_omni.stages import load_minimind_omni_bundle

            extra = dict(self.engine_args.extra or {})
            mimi_model_id = extra.pop("mimi_model_id", None) or extra.pop("mimi", None)
            kwargs = {}
            if mimi_model_id:
                kwargs["mimi_model_id"] = mimi_model_id
            self._bundle = load_minimind_omni_bundle(
                model_id=self.model,
                device=self.engine_args.device,
                **kwargs,
            )
        return self._bundle
