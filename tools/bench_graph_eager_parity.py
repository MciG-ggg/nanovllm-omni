"""Compare graph vs eager token counts + audio parity on the thinker stage.

Run on WSL RTX 3050::

    cd ~/nanovllm-omni && source ~/venvs/nanovllm-omni/bin/activate
    HF_HUB_OFFLINE=1 PYTHONPATH=. python tools/bench_graph_eager_parity.py \\
        --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \\
        --prompts "Hi" "Hello there" "Tell me a short story" \\
        --max-tokens 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/mcig/minimind-3o")
    parser.add_argument("--mimi", default="/home/mcig/mimi")
    parser.add_argument(
        "--prompts", nargs="+", default=["Hi", "Hello there", "Tell me a short story"]
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("No CUDA.")
        return 1

    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle
    from nanovllm_omni.models.minimind_omni.generation import stream_generate
    from nanovllm_omni.optim.cuda_graph import enable_cuda_graph

    bundle = load_minimind_omni_bundle(model_id=args.model, mimi_model_id=args.mimi)
    model = bundle.model
    tok = bundle.tokenizer

    rows = []
    for prompt in args.prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]

        # Eager
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eager_text: list[int] = []
        eager_audio_frames = 0
        for _text, audio in stream_generate(
            model,
            ids,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=getattr(model, "eos_token_id_2", 2),
            open_thinking=False,
        ):
            if _text is not None:
                eager_text = list(_text.reshape(-1).tolist())
            if audio is not None:
                eager_audio_frames += 1
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000.0

        # Graph
        decoder = enable_cuda_graph(
            model, n_steps=args.max_tokens, max_len=plen + args.max_tokens + 4
        )
        assert decoder is not None
        decoder.temperature = 0.7
        decoder.top_p = 0.9
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g_out = decoder.generate_tokens(ids, seed=args.seed, return_audio=True)
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1000.0
        g_text, g_audio = g_out
        graph_audio_frames = sum(
            1 for layer in g_audio if any(c != decoder.audio_pad for c in layer)
        )

        rows.append(
            {
                "prompt": prompt,
                "prompt_len": plen,
                "eager_ms": eager_ms,
                "eager_text_tokens": len(eager_text),
                "eager_audio_frames": eager_audio_frames,
                "graph_ms": graph_ms,
                "graph_text_tokens": len(g_text),
                "graph_audio_nonpad": graph_audio_frames,
                "speedup": eager_ms / max(graph_ms, 1.0),
            }
        )

    print(json.dumps(rows, indent=2))

    avg_speedup = sum(r["speedup"] for r in rows) / max(len(rows), 1)
    print(f"\nAverage speedup over eager: {avg_speedup:.2f}x")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
