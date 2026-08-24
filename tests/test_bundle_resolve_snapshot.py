"""Contract tests for ``bundle._resolve_snapshot``.

Mirrors vllm-omni's ``_resolve_model_to_local_path`` semantics:
a path on disk is used as-is, anything else is looked up in the
HuggingFace local cache with ``local_files_only=True`` (never triggers
a network download), unresolvable Hub ids are passed through unchanged
with a warning. These tests pin the offline-first behavior so the
regression that re-enables network auto-download is caught early.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def any_tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# local directory
# ---------------------------------------------------------------------------


def test_resolve_snapshot_uses_local_directory_verbatim(any_tmp_dir):
    """A path that already exists is returned resolved; no HF cache lookup."""
    from nanovllm_omni.models.minimind_omni.bundle import _resolve_snapshot

    out = _resolve_snapshot(str(any_tmp_dir))
    assert out == str(any_tmp_dir.resolve())
    # Critically: the function returns before snapshot_download is touched,
    # so even a non-existent HF cache cannot fail this path.


# ---------------------------------------------------------------------------
# Hub id with local cache hit
# ---------------------------------------------------------------------------


def test_resolve_snapshot_returns_local_files_only_snapshot_when_cached(
    monkeypatch,
):
    """Hub id with a snapshot present in ``HF_HUB_CACHE`` returns it
    via ``snapshot_download(..., local_files_only=True)`` -- the resolved
    path equals what the cache returns and is never the bare Hub id."""
    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    sentinel = "/fake/hf/cache/models--org--name/snapshots/abc123"
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda model_id, local_files_only: (
            sentinel if local_files_only else pytest.fail("must be local_files_only=True")
        ),
    )

    out = bundle_mod._resolve_snapshot("org/name")
    assert out == sentinel


# ---------------------------------------------------------------------------
# Hub id with no local cache -- offline-first, no network, warning + passthrough
# ---------------------------------------------------------------------------


def test_resolve_snapshot_does_not_network_when_no_local_cache(monkeypatch, caplog):
    """When ``snapshot_download(local_files_only=True)`` raises (no local
    cache, offline), the function logs a warning and returns the original
    Hub id unchanged. ``snapshot_download`` must NOT have been called with
    ``local_files_only=False`` -- that would silently re-enable network."""
    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    seen: dict[str, object] = {}

    def fake_snapshot(model_id, local_files_only=True):
        seen["model_id"] = model_id
        seen["local_files_only"] = local_files_only
        from huggingface_hub.errors import LocalEntryNotFoundError

        raise LocalEntryNotFoundError(
            "The model does not exist locally and we cannot fetch it offline."
        )

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    with caplog.at_level(logging.WARNING, logger="nanovllm_omni.models.minimind_omni.bundle"):
        out = bundle_mod._resolve_snapshot("jingyaogong/minimind-3o")

    # Returns the original id unchanged instead of raising.
    assert out == "jingyaogong/minimind-3o"
    # local_files_only=True was honored -- no network path.
    assert seen == {
        "model_id": "jingyaogong/minimind-3o",
        "local_files_only": True,
    }
    # And a warning was emitted naming the offline-first passthrough.
    assert any(
        "Could not resolve" in rec.message and "jingyaogong/minimind-3o" in rec.message
        for rec in caplog.records
    )


def test_resolve_snapshot_handles_arbitrary_exception(monkeypatch, caplog):
    """Any other exception (network is unreachable, etc.) is treated the
    same as a missing local cache: warning + passthrough. We never propagate
    the exception to the caller of ``_resolve_snapshot`` -- the user gets
    a clear pass-through so a downstream ``from_pretrained`` can surface
    a better error if it must."""
    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    def boom(model_id, local_files_only=True):
        raise ConnectionError("[Errno 101] Network is unreachable")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)

    with caplog.at_level(logging.WARNING, logger="nanovllm_omni.models.minimind_omni.bundle"):
        out = bundle_mod._resolve_snapshot("any/hub-id")

    assert out == "any/hub-id"
    assert any(
        "Could not resolve" in rec.message and "ConnectionError" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Local directory short-circuits BEFORE snapshot_download
# ---------------------------------------------------------------------------


def test_resolve_snapshot_local_directory_skips_snapshot_download(monkeypatch, any_tmp_dir):
    """When ``model_id`` is a real local directory, ``snapshot_download``
    must not be touched at all -- otherwise we'd risk running an HF lookup
    against arbitrary paths the user passes in."""
    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    called = {"hit": False}
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: called.update(hit=True),
    )

    out = bundle_mod._resolve_snapshot(str(any_tmp_dir))
    assert out == str(any_tmp_dir.resolve())
    assert called["hit"] is False


__all__ = [
    "test_resolve_snapshot_uses_local_directory_verbatim",
    "test_resolve_snapshot_returns_local_files_only_snapshot_when_cached",
    "test_resolve_snapshot_does_not_network_when_no_local_cache",
    "test_resolve_snapshot_handles_arbitrary_exception",
    "test_resolve_snapshot_local_directory_skips_snapshot_download",
]
