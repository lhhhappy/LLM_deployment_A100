#!/usr/bin/env python3
"""Check that vLLM tokenizes every replayed prompt to its frozen ``glm_tokens``.

The load generator renders each request body with the harness ``Renderer``
(chat template, no tokenization) and posts the text to ``/generate``. The
``generate_compat`` route tokenizes that text with the completion renderer
vLLM builds for the model. This script runs both steps on CPU for every request
of a data set and compares the token count with the frozen ``glm_tokens``.
Exit status 0 means every request matches.

Run with the vLLM environment, e.g.
  python scripts/vllm/check_prompt_tokens.py --data-root data/s1-dev-longchain \
      --model-dir s1-dev/glm_tok --out evidence/vllm-tokens/<name>.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "s1-dev", "harness"))

_worker = {}


def _init_worker(model_dir: str) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers import renderer_from_config
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.usage.usage_lib import UsageContext

    vllm_config = EngineArgs(
        model=model_dir, tokenizer=model_dir, load_format="dummy"
    ).create_engine_config(UsageContext.OPENAI_API_SERVER)
    _worker["renderer"] = renderer_from_config(vllm_config)
    _worker["model_config"] = vllm_config.model_config
    _worker["parse"] = parse_model_prompt


def _count(text: str) -> int:
    prompt = _worker["parse"](_worker["model_config"], {"prompt": text})
    (engine_input,) = _worker["renderer"].render_cmpl([prompt])
    return len(engine_input["prompt_token_ids"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-dir", required=True, help="config.json + tokenizer files")
    ap.add_argument("--out", required=True, help="JSON summary path")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N requests (0 = all)")
    args = ap.parse_args()

    from s1_common import Renderer, load_index, materialize_bodies

    rows, _chains, _by_chain = load_index(args.data_root)
    rids = sorted(rows)
    if args.limit:
        rids = rids[: args.limit]
    bodies = materialize_bodies(args.data_root, rids)
    missing = [r for r in rids if r not in bodies]
    harness = Renderer(args.model_dir)
    texts = [harness.render(bodies[r]) for r in rids if r in bodies]
    kept = [r for r in rids if r in bodies]

    t0 = time.time()
    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(args.model_dir,)
    ) as pool:
        counts = list(pool.map(_count, texts, chunksize=8))
    elapsed = time.time() - t0

    mismatches = [
        {"req_id": r, "vllm": n, "glm_tokens": rows[r]["glm_tokens"]}
        for r, n in zip(kept, counts)
        if n != rows[r]["glm_tokens"]
    ]
    manifest = os.path.join(args.data_root, "manifest.json")
    summary = {
        "data_root": args.data_root,
        "manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest(),
        "model_dir": args.model_dir,
        "requests": len(rids),
        "bodies_missing": len(missing),
        "checked": len(kept),
        "mismatched": len(mismatches),
        "total_tokens": sum(counts),
        "elapsed_s": round(elapsed, 1),
        "first_mismatches": mismatches[:20],
        "verdict": "PASS" if not mismatches and not missing else "FAIL",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "first_mismatches"}))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
