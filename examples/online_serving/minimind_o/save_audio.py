"""Decode ``choices[0].message.audio.data`` from a
``/v1/chat/completions`` response into a WAV file.

Stdlib only (``json``, ``base64``) so it works on any machine that can
reach the server. Pipe or save the curl output, then::

    bash curl_chat.sh "你好" > resp.json
    python save_audio.py resp.json --out hello.wav
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", help="JSON file with the chat completion response")
    parser.add_argument("--out", default="response.wav", help="output wav path")
    args = parser.parse_args()

    body = json.loads(Path(args.response).read_text())
    try:
        audio = body["choices"][0]["message"]["audio"]
        b64 = audio["data"]
        sample_rate = audio.get("sample_rate", 24000)
    except (KeyError, IndexError, TypeError) as exc:
        print(f"[save_audio] FAIL: malformed response: {exc}", file=sys.stderr)
        return 1

    raw = base64.b64decode(b64)
    Path(args.out).write_bytes(raw)
    print(f"wrote {args.out} ({len(raw)} bytes, sr={sample_rate})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
