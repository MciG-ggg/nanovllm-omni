# docs/perf/archive — historical session records

Point-in-time lab records from the MiniMind-O optimization journey
(sessions 5–12). Most cover experiments that have since been **removed
from the runtime** (`--compile` / `--int8` / `--graph` bench flags and
the `optim/compile.py` / `optim/cuda_graph.py` modules were deleted in
commit `7ac8bf3` — they were never real wins, see the canonical doc).

**Current truth is NOT here.** Read:

- `../minimind-omni-under-500ms.md` — the consolidated optimization story
  (E1–E25 experiment table, the 13 monkey-patches, lessons learned).
- `../session-1.md` — the current MiniMind-O baseline (RTX 3050 4 GB).

These files are gitignored (`docs/perf/*` in `.gitignore`); they were
never committed to the repository. Keep them as local reference;
re-run `python -m nanovllm_omni.optim.bench time` to produce fresh
numbers instead of trusting these.