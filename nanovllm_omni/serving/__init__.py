"""Gradio serving layer.

app.py   -- entry point; lazy-loads models per tab (HF Spaces friendly)
audio.py -- MiniMind-O TTS tab
chat.py  -- SmolVLM-500M chat tab
image.py -- SD-Turbo text-to-image tab
vla.py   -- SmolVLA action-chunk tab
"""

from .app import build_demo, main

__all__ = ["build_demo", "main"]
