"""Build the n-gram draft table for native speculative decoding (`BABBLE_NATIVE_SPEC=1`).

The table maps the last 1-3 tokens to the most likely next token (or "don't
draft" when the top continuation is below --min-conf). Sources, both
consent-free:

* a random-offset sample of mookiezi/Discord-Dialogues (the SFT data) fetched
  through the HF datasets-server rows API -- 100 dialogs per request, nothing
  like the 347 MB parquet;
* replies sampled from the served model itself on those prompts (its own
  phrasing, at the live sampling settings).

    python bench/extreme/spec_ngram.py fetch  --out DIR --requests 120
    python bench/extreme/spec_ngram.py sample --dd DIR --model MODEL_DIR --out samples.jsonl --prompts 2000
    python bench/extreme/spec_ngram.py build  --dd DIR --samples samples.jsonl --model MODEL_DIR --out spec-ngram.pt

Point `BABBLE_NATIVE_SPEC_TABLE` at the result (or drop it in the model dir as
`spec-ngram.pt`). It is a derived artifact: keep it out of git like the weights.
Never build it from the bot's consented corpus or logs.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import subprocess
import time
from pathlib import Path

TURN = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", flags=re.S)


def fetch(out: Path, requests: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(requests):
        path = out / f"r{i}.json"
        if path.exists() and path.stat().st_size > 1000:
            continue
        off = (i * 24337 + 1234567) % 7300800
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=mookiezi/Discord-Dialogues"
            f"&config=default&split=train&offset={off}&length=100"
        )
        for _ in range(3):  # curl, not urllib: some fronts 403 the Python-urllib UA
            r = subprocess.run(["curl", "-s", "-m", "30", "-A", "babble-spec-ngram/1", url], capture_output=True)
            try:
                rows = json.loads(r.stdout)["rows"]
                path.write_text(json.dumps([row["row"]["text"] for row in rows]))
                break
            except (ValueError, KeyError):
                time.sleep(20)  # rate limited
        time.sleep(3)


def dialogs(dd: Path) -> list[list[tuple[str, str]]]:
    files = sorted(glob.glob(str(dd / "r*.json")), key=lambda p: int(re.findall(r"\d+", Path(p).name)[0]))
    return [TURN.findall(text) for f in files for text in json.loads(Path(f).read_text())]


def sample(dd: Path, model: Path, out: Path, prompts: int, seed: int) -> None:
    from tokenizers import Tokenizer

    from babble.leanserve import SamplingConfig
    from babble.nativeserve import NativeEngine

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    cfg = SamplingConfig(temperature=0.5, top_k=40, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4)
    eng = NativeEngine(model, threads=4)
    rng = random.Random(seed)
    with out.open("w") as f:
        for i, turns in enumerate(d for d in dialogs(dd) if d):
            if i >= prompts:
                break
            prior = turns[: rng.randint(1, len(turns))]
            last = prior[-1][0]
            text = "\n".join(f"{'user' if r == last else 'assistant'}: {b.strip()}" for r, b in prior[-17:])
            ids = [bos, *tok.encode(text, add_special_tokens=False).ids[-1536:], sep]
            o = eng.generate(ids, n=4, max_new=128, sampling=cfg, eos_id=eos, seed=seed + i)
            f.write(json.dumps({"prompt": ids, "cands": o.tokens}) + "\n")


def build(dd: Path | None, samples: Path | None, model: Path, out: Path, min_conf: float, order: int) -> None:
    from tokenizers import Tokenizer

    from babble.nativeserve import build_ngram_tables, save_ngram_tables

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    sep, eos, vocab = tok.token_to_id("<sep>"), tok.token_to_id("<eos>"), tok.get_vocab_size()
    seqs = []
    if dd:  # every turn as a reply: <sep> body <eos>
        for turns in dialogs(dd):
            seqs += [[sep, *tok.encode(b.strip(), add_special_tokens=False).ids, eos] for _r, b in turns]
    if samples:
        for line in samples.open():
            row = json.loads(line)
            seqs += [[sep, *c] for c in row["cands"]]
    tables = build_ngram_tables(seqs, order=order, min_conf=min_conf, vocab=vocab)
    meta = {"min_conf": min_conf, "order": order, "seqs": len(seqs), "tokens": sum(map(len, seqs))}
    save_ngram_tables(tables, out, vocab=vocab, meta=meta)
    print(json.dumps({**meta, "contexts": {n: len(t) for n, t in tables.items()}}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--requests", type=int, default=120)
    s = sub.add_parser("sample")
    s.add_argument("--dd", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--prompts", type=int, default=2000)
    s.add_argument("--seed", type=int, default=1234)
    b = sub.add_parser("build")
    b.add_argument("--dd", type=Path)
    b.add_argument("--samples", type=Path)
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--min-conf", type=float, default=0.5)
    b.add_argument("--order", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.out, a.requests)
    elif a.cmd == "sample":
        sample(a.dd, a.model, a.out, a.prompts, a.seed)
    else:
        build(a.dd, a.samples, a.model, a.out, a.min_conf, a.order)


if __name__ == "__main__":
    main()
