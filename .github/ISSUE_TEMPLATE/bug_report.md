---
name: Bug report
about: Report a bug or unexpected behaviour
title: "bug: "
labels: ["bug"]
assignees: []
---

## What happened

A short, concrete description of the bug. One sentence is enough.

## Reproduction

Minimal code or shell that reproduces the issue:

```python
from nanovllm_omni import Omni, SamplingParams

engine = Omni("...")
outputs = engine.generate(["..."], SamplingParams(...))
# actual behaviour:
```

If the bug is in the HTTP seam (`POST /v1/chat/completions`), include
the request and the full JSON response.

## Expected behaviour

What you expected to happen instead, and why.

## Environment

- `nanovllm-omni` version (output of `python -c "import nanovllm_omni; print(nanovllm_omni.__version__)"` or `pip show nanovllm-omni`)
- Python version (`python --version`)
- OS / hardware (e.g. macOS 14 / M2, Ubuntu 22.04 / RTX 3050)
- Model family + weight checkpoint id / local path
- Install extras used (`.[dev]`, `.[dev,minimind]`, `.[smolvla]`, etc.)

## Logs / stack trace

Paste the relevant log lines or the full traceback. Trim unrelated
output.
