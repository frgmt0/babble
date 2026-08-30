"""Long-form SFT for Booper-Big-Chat on a laptop (MPS / CPU / CUDA).

The live model (`ProCreations/Booper-Big-Chat-INT8`) was SFT'd only on
`mookiezi/Discord-Dialogues`, whose replies are one-liners, so after `<sep>`
its prior is "say a few tokens, emit <eos>". This script continues the SFT on
a mix of long-form pairs (TinyStories-Instruct stories, no_robots answers)
plus a Discord-Dialogues rehearsal slice so the chat voice survives, all in
the pair layout the bot serves: `<bos> prompt <sep> response <eos>`, loss on
the response only.

Outputs land in `runs/<name>/`:
  metrics.jsonl   one JSON line per log step (loss, lr, tok/s, val, samples)
  train.log       stdout of the run (what `monitor.sh` tails)
  ckpt/           latest bf16 safetensors + config + tokenizer (resumable)
  export/         INT8 pack in the exact layout `babble.hfserve._load_int8`
                  reads (`--export` or automatic at the end)

Usage (from the repo root, venv active):
  python sft/sft_longform.py --name story-v1 --tokens 30e6
  python sft/sft_longform.py --name story-v1 --resume
  python sft/sft_longform.py --name story-v1 --export   # only re-pack
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INT8_REPO = "ProCreations/Booper-Big-Chat-INT8"
SAMPLE_PROMPTS = [
    "write me a short story about a dragon who is afraid of fire",
    "write a short story about a detective in a city that never sleeps",
    "hey booper whats up",
]
STORY_TEMPLATES = [
    "write me a story about {summary}",
    "can you write a short story? {summary}",
    "tell me a story where {summary}",
    "write a story using the words {words}",
    "story time! something about {summary}",
    "write a short story with these words: {words}",
]


# ----------------------------------------------------------------- data ---


def _tinystories_records(split: str, seed: int):
    """Yield (prompt, story) from TinyStoriesInstruct's line-per-row layout."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True)
    fields: dict[str, str] = {}
    story: list[str] = []
    in_story = False
    for row in ds:
        line = row["text"]
        if line == "<|endoftext|>":
            text = "\n".join(story).strip()
            if text and (fields.get("Summary") or fields.get("Words")):
                summary = fields.get("Summary", "").strip().rstrip(".")
                words = fields.get("Words", "").strip()
                if summary:
                    summary = summary[0].lower() + summary[1:]
                tmpl = rng.choice(STORY_TEMPLATES)
                if "{words}" in tmpl and not words:
                    tmpl = STORY_TEMPLATES[0]
                if "{summary}" in tmpl and not summary:
                    tmpl = STORY_TEMPLATES[3]
                yield tmpl.format(summary=summary, words=words), text
            fields, story, in_story = {}, [], False
            continue
        if in_story:
            story.append(line)
        elif line.startswith("Story:"):
            in_story = True
        elif ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()


def _no_robots_records(split: str):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/no_robots", split=split)
    for row in ds:
        msgs = row["messages"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            yield msgs[0]["content"].strip(), msgs[1]["content"].strip()


_WP_FIXES = [
    (r"\s+([,.!?;:%])", r"\1"), (r"\s+'\s*(s|t|re|ve|ll|d|m)\b", r"'\1"), (r"\bn't\b", "n't"),
    (r"``\s*", '"'), (r"\s*''", '"'), (r"\(\s+", "("), (r"\s+\)", ")"), (r"\s+n't", "n't"),
    (r"([\"])\s+([^\"]*?)\s+([\"])", r"\1\2\3"), (r" {2,}", " "),
]


def _wp_clean(text: str) -> str:
    """Undo writingprompts' PTB-style tokenisation ("You 've", "`` quote '')."""
    import re

    text = re.sub(r"^\s*\[\s*[A-Z]{2,3}\s*\]\s*", "", text)  # [ WP ] / [ EU ] / [ TT ] tags
    for pat, rep in _WP_FIXES:
        text = re.sub(pat, rep, text)
    return text.strip()


def _writingprompts_records(split: str, max_chars: int = 3500):
    """r/WritingPrompts prompt -> story, adult register. Long ones are skipped, not cut."""
    from datasets import load_dataset

    ds = load_dataset("euclaise/writingprompts", split=split, streaming=True)
    for row in ds:
        story = row["story"]
        if len(story) > max_chars or len(story) < 200:
            continue
        prompt = _wp_clean(row["prompt"])
        if not prompt:
            continue
        yield prompt, _wp_clean(story)


def _smoltalk_records(split: str, configs=("smol-magpie-ultra", "everyday-conversations")):
    """First user->assistant turn of SmolTalk's general-assistant subsets."""
    from datasets import load_dataset

    for cfg in configs:
        ds = load_dataset("HuggingFaceTB/smoltalk", cfg, split=split, streaming=True)
        for row in ds:
            msgs = [m for m in row["messages"] if m["role"] in ("user", "assistant")]
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                yield msgs[0]["content"].strip(), msgs[1]["content"].strip()


def _discord_records(split: str):
    """Last user->assistant pair of each Discord-Dialogues conversation."""
    from datasets import load_dataset

    ds = load_dataset("mookiezi/Discord-Dialogues", split=split, streaming=True)
    for row in ds:
        turns = []
        for chunk in row["text"].split("<|im_start|>")[1:]:
            role, _, body = chunk.partition("\n")
            body = body.split("<|im_end|>")[0].strip()
            turns.append((role.strip(), body))
        for i in range(len(turns) - 1, 0, -1):
            if turns[i][0] == "assistant" and turns[i - 1][0] == "user" and turns[i][1]:
                yield turns[i - 1][1], turns[i][1]
                break


def build_examples(tok, args, log):
    """Tokenize a mixed, shuffled list of (ids, n_prompt) pairs plus a val slice."""
    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    def repeated(gen, times):
        def run():
            for _ in range(times):
                yield from gen()
        return run

    sources = [
        ("tinystories", args.mix_story, lambda: _tinystories_records("train", args.seed)),
        ("writingprompts", args.mix_wp, lambda: _writingprompts_records("train")),
        ("no_robots", args.mix_norobots, repeated(lambda: _no_robots_records("train"), args.repeat_norobots)),
        ("smoltalk", args.mix_smoltalk, lambda: _smoltalk_records("train")),
        ("discord", args.mix_discord, lambda: _discord_records("train")),
    ]
    total = sum(w for _, w, _ in sources)
    examples: list[tuple[list[int], int]] = []
    counts = {}
    for name, weight, gen in sources:
        if weight <= 0:
            continue
        want = int(args.examples * weight / total)
        got = 0
        for prompt, response in gen():
            p = tok.encode(prompt, add_special_tokens=False).ids[-args.prompt_budget :]
            r = tok.encode(response, add_special_tokens=False).ids
            if len(r) < args.min_response or len(p) + len(r) + 3 > args.seq_len:
                continue
            examples.append(([bos, *p, sep, *r, eos], len(p) + 2))
            got += 1
            if got >= want:
                break
        counts[name] = got
        log(f"data: {name} -> {got} examples")
    random.Random(args.seed).shuffle(examples)
    n_val = min(args.val_examples, len(examples) // 20)
    return examples[n_val:], examples[:n_val], counts


def batches(examples, tokens_per_batch, pad_id, shuffle_seed=None):
    """Length-bucketed batches under a token budget (padding counted)."""
    order = list(range(len(examples)))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(order)
        # Sort within wide windows so padding stays small but order stays random.
        window = 256
        chunks = [order[i : i + window] for i in range(0, len(order), window)]
        order = [i for c in chunks for i in sorted(c, key=lambda j: len(examples[j][0]))]
        groups = []
        cur, cur_max = [], 0
        for i in order:
            n = len(examples[i][0])
            if cur and max(cur_max, n) * (len(cur) + 1) > tokens_per_batch:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
        rng.shuffle(groups)
    else:
        order.sort(key=lambda j: len(examples[j][0]))
        groups, cur, cur_max = [], [], 0
        for i in order:
            n = len(examples[i][0])
            if cur and max(cur_max, n) * (len(cur) + 1) > tokens_per_batch:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
    for g in groups:
        width = max(len(examples[i][0]) for i in g)
        ids = torch.full((len(g), width), pad_id, dtype=torch.long)
        labels = torch.full((len(g), width), -100, dtype=torch.long)
        for r, i in enumerate(g):
            toks, n_prompt = examples[i]
            ids[r, : len(toks)] = torch.tensor(toks)
            labels[r, n_prompt : len(toks)] = torch.tensor(toks[n_prompt:])
        yield ids, labels


# ---------------------------------------------------------------- model ---


def load_base(model_dir: Path, device, log):
    from babble.hfserve import _load_int8

    model, config = _load_int8(model_dir)
    model.train()
    log(f"model: {sum(p.numel() for p in model.parameters()):,} params from {model_dir}")
    return model.to(device), config


def fetch_base(cache: Path, log) -> Path:
    from huggingface_hub import snapshot_download

    if (cache / "model-int8.safetensors").exists():
        return cache
    log(f"fetching {INT8_REPO} -> {cache}")
    return Path(snapshot_download(INT8_REPO, local_dir=str(cache)))


def save_ckpt(model, config, tok_path: Path, out: Path, step: int, tokens: int, opt=None):
    from safetensors.torch import save_file

    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    # NOTE: `.to("cpu", torch.bfloat16)` in ONE call corrupts the source fp32
    # tensor on MPS (torch 2.13) -- measured: val 2.33 -> 4.12 from that line
    # alone. Copy to CPU first, cast second. Same in export_int8.
    state = {k: v.detach().to("cpu").to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    if "lm_head.weight" in state and getattr(config, "tie_word_embeddings", False):
        del state["lm_head.weight"]  # tied; re-tied on load
    save_file(state, str(tmp / "model.safetensors"))
    config.save_pretrained(tmp)
    shutil.copy(tok_path, tmp / "tokenizer.json")
    (tmp / "state.json").write_text(json.dumps({"step": step, "tokens": tokens}))
    if opt is not None:
        torch.save(opt.state_dict(), tmp / "optim.pt")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def load_ckpt(model, ckpt: Path, device, opt=None):
    from safetensors.torch import load_file

    state = load_file(str(ckpt / "model.safetensors"))
    missing, unexpected = model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=False)
    assert not unexpected and all(m == "lm_head.weight" for m in missing), (missing, unexpected)
    model.tie_weights()
    meta = json.loads((ckpt / "state.json").read_text())
    if opt is not None and (ckpt / "optim.pt").exists():
        opt.load_state_dict(torch.load(ckpt / "optim.pt", map_location=device))
    return meta["step"], meta["tokens"]


def export_int8(model, config, tok_path: Path, src_dir: Path, out: Path, log):
    """Re-pack into the per-output-channel symmetric INT8 layout of the live snapshot."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to("cpu").to(torch.float32) for k, v in model.state_dict().items()}
    fused = any(".mlp.experts.gate_up_proj" in k for k in sd)
    packed: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}

    def q(name: str, w: torch.Tensor):
        if w.dim() == 1:
            packed[name] = w.to(torch.bfloat16)
            return
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        packed[name] = torch.round(w / scale).clamp(-127, 127).to(torch.int8).contiguous()
        packed[name + ".scale"] = scale.to(torch.bfloat16).contiguous()
        report[name] = {"axis": 1, "dtype": "int8", "scale": name + ".scale"}

    for name, w in sd.items():
        if name == "lm_head.weight" and "model.embed_tokens.weight" in sd:
            q(name, w)  # stored explicitly in the original artifact too
            continue
        if ".mlp." in name and fused:
            continue
        if ".block_sparse_moe." in name:
            q(name, w)
            continue
        q(name, w)
    if "lm_head.weight" not in sd:
        q("lm_head.weight", sd["model.embed_tokens.weight"])
    if fused:
        inter = config.intermediate_size
        for layer in range(config.num_hidden_layers):
            new = f"model.layers.{layer}.mlp"
            old = f"model.layers.{layer}.block_sparse_moe"
            q(old + ".gate.weight", sd[new + ".gate.weight"])
            gate_up = sd[new + ".experts.gate_up_proj"]
            down = sd[new + ".experts.down_proj"]
            for e in range(config.num_local_experts):
                q(f"{old}.experts.{e}.w1.weight", gate_up[e, :inter])
                q(f"{old}.experts.{e}.w3.weight", gate_up[e, inter:])
                q(f"{old}.experts.{e}.w2.weight", down[e])
    save_file(packed, str(out / "model-int8.safetensors"))
    config.save_pretrained(out)
    shutil.copy(tok_path, out / "tokenizer.json")
    for extra in ("tokenizer_config.json", "load_int8.py"):
        if (src_dir / extra).exists():
            shutil.copy(src_dir / extra, out / extra)
    (out / "quantization_config.json").write_text(
        json.dumps(
            {
                "activations": "bfloat16",
                "granularity": "per-output-channel",
                "quant_method": "booper_symmetric_int8",
                "source": "sft/sft_longform.py",
                "tensors": report,
            },
            indent=2,
        )
    )
    log(f"export: {out / 'model-int8.safetensors'} ({(out / 'model-int8.safetensors').stat().st_size / 1e6:.1f} MB)")
    # Round-trip through the serving loader so a bad pack fails here, not live.
    from babble.hfserve import _load_int8

    _load_int8(out)
    log("export: round-trip load through babble.hfserve._load_int8 OK")


def push_export(run_dir: Path, repo: str, log):
    """Upload runs/<name>/export (plus a model card built from metrics.jsonl) to the Hub."""
    from huggingface_hub import HfApi

    export = run_dir / "export"
    recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").open()] if (run_dir / "metrics.jsonl").exists() else []
    start = next((r for r in recs if r.get("event") == "start"), {})
    evals = [r for r in recs if "val" in r]
    trains = [r for r in recs if "loss" in r]
    samples = evals[-1].get("samples", []) if evals else []
    card = [
        "---", "license: apache-2.0", "language: [en]", "library_name: transformers", "pipeline_tag: text-generation",
        f"base_model: {INT8_REPO}", "tags: [moe, booper, int8, sft, long-form]", "---", "",
        f"# {repo.split('/')[-1]}", "",
        f"Long-form SFT of `{INT8_REPO}` (Mixtral MoE, 150M total / ~50M active, vocab 16384) so booper answers",
        "story/long-answer requests instead of one-liners. Trained with `sft/sft_longform.py` from",
        "https://github.com/frgmt0/babble in the pair layout `<bos> prompt <sep> response <eos>` (loss on the response).", "",
        "## Data mix", "",
        *(f"- {k}: {v} examples" for k, v in (start.get("counts") or {}).items()),
        "", "## Training", "",
        f"- steps: {trains[-1]['step'] if trains else '?'}, tokens: {trains[-1]['tokens']:,}" if trains else "- (no train records)",
        f"- val loss: {evals[0]['val']:.4f} -> {evals[-1]['val']:.4f}" if len(evals) > 1 else "",
        f"- device: {start.get('device', '?')}, lr {start.get('args', {}).get('lr')}, seq len {start.get('args', {}).get('seq_len')}",
        "", "## Samples (temperature 0.8)", "",
        *(f"**{s['prompt']}**\n\n> {s['reply'].replace(chr(10), chr(10) + '> ')}\n" for s in samples),
        "", "## Loading", "",
        "Same INT8 layout as the base: `load_int8.py` in this repo, or `babble.hfserve` with `BABBLE_HF_MODEL_DIR` pointed at a snapshot.",
    ]
    (export / "README.md").write_text("\n".join(c for c in card if c is not None))
    shutil.copy(run_dir / "metrics.jsonl", export / "metrics.jsonl")
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    log(f"push: uploading {export} -> https://huggingface.co/{repo}")
    api.upload_folder(folder_path=str(export), repo_id=repo, commit_message=f"SFT run {run_dir.name}")
    log(f"push: done https://huggingface.co/{repo}")


# ------------------------------------------------------------ training ---


@torch.no_grad()
def free_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)


def sample(model, tok, device, max_new=120):
    bos, sep, eos, pad = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>", "<pad>"))
    model.eval()
    outs = []
    for p in SAMPLE_PROMPTS:
        ids = torch.tensor([[bos, *tok.encode(p, add_special_tokens=False).ids, sep]], device=device)
        # Two draws at a cooler temperature than serving (0.5 vs 0.8) so a
        # sample says something about the weights, not the dice; no_repeat_ngram
        # matches what live serves.
        gen = model.generate(
            ids, do_sample=True, temperature=0.5, top_p=0.95, max_new_tokens=max_new, num_return_sequences=2,
            eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.1, no_repeat_ngram_size=4,
        )[:, ids.shape[1] :]
        for row in gen:
            keep = [int(t) for t in row if int(t) not in (pad, eos)]
            outs.append({"prompt": p, "reply": tok.decode(keep, skip_special_tokens=True).strip(), "tokens": len(keep)})
        del gen
        free_cache(device)
    model.train()
    return outs


@torch.no_grad()
def evaluate(model, val, device, pad, dtype):
    model.eval()
    tot, n = 0.0, 0
    for ids, labels in batches(val, 2048, pad):
        ids, labels = ids.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(input_ids=ids, attention_mask=(ids != pad)).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot += float(loss)
        n += int((labels[:, 1:] != -100).sum())
        del logits, loss
    free_cache(device)
    model.train()
    return tot / max(n, 1)


class Reporter:
    """Append to metrics.jsonl and optionally POST each record to the /runs endpoint."""

    def __init__(self, run_dir: Path, name: str):
        self.path = run_dir / "metrics.jsonl"
        self.name = name
        self.url = os.environ.get("BABBLE_RUNS_URL")
        self.token = os.environ.get("BABBLE_RUNS_TOKEN")

    def __call__(self, rec: dict):
        rec = {"run": self.name, "t": time.time(), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.url and self.token:
            try:
                req = urllib.request.Request(
                    f"{self.url.rstrip('/')}/api/runs/{self.name}",
                    data=json.dumps(rec).encode(),
                    # Cloudflare's bot rules 403 the default "Python-urllib" agent.
                    headers={"content-type": "application/json", "authorization": f"Bearer {self.token}", "user-agent": "babble-sft/1.0"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as e:  # never let the dashboard kill a run
                print(f"reporter: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=float, default=30e6, help="training-token budget (input tokens incl. prompt)")
    ap.add_argument("--examples", type=int, default=120_000, help="examples to tokenize across the mix")
    ap.add_argument("--mix-story", type=float, default=0.45, help="roneneldan/TinyStoriesInstruct")
    ap.add_argument("--mix-wp", type=float, default=0.0, help="euclaise/writingprompts (adult-register fiction)")
    ap.add_argument("--mix-norobots", type=float, default=0.15, help="HuggingFaceH4/no_robots")
    ap.add_argument("--repeat-norobots", type=int, default=1, help="no_robots is only ~8.4k pairs; epochs to upsample it")
    ap.add_argument("--mix-smoltalk", type=float, default=0.0, help="HuggingFaceTB/smoltalk general subsets")
    ap.add_argument("--mix-discord", type=float, default=0.40, help="mookiezi/Discord-Dialogues rehearsal")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--prompt-budget", type=int, default=256)
    ap.add_argument("--min-response", type=int, default=1)
    ap.add_argument("--val-examples", type=int, default=400)
    ap.add_argument("--tokens-per-batch", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", default=None, help="dir with model-int8.safetensors (default: fetch to artifacts/)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", action="store_true", help="only re-pack runs/<name>/ckpt to INT8")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 600 examples, 12 steps")
    ap.add_argument("--push", default=None, metavar="NAMESPACE/REPO", help="after export, upload runs/<name>/export to this HF repo (uses the cached HF login)")
    args = ap.parse_args()
    if args.smoke:
        args.examples, args.tokens, args.log_every, args.eval_every, args.ckpt_every = 600, 12 * args.tokens_per_batch * args.accum, 2, 6, 6

    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "train.log").open("a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    # Bound the MPS allocator to a fraction of unified memory so a leak fails
    # loudly instead of paging the whole machine (default 1.0 = "everything").
    # Both must be set: PyTorch's default LOW is 1.4 and it rejects HIGH < LOW.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    log(f"device={device} autocast={dtype}")

    base = Path(args.base) if args.base else fetch_base(ROOT / "artifacts" / "hf-booper-big-chat-int8", log)
    from tokenizers import Tokenizer

    tok_path = base / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    pad = tok.token_to_id("<pad>")
    model, config = load_base(base, device, log)
    ckpt_dir = run_dir / "ckpt"

    if args.export:
        step, tokens = load_ckpt(model, ckpt_dir, device)
        log(f"export from step {step} ({tokens:,} tokens)")
        export_int8(model, config, tok_path, base, run_dir / "export", log)
        if args.push:
            push_export(run_dir, args.push, log)
        return

    train, val, counts = build_examples(tok, args, log)
    log(f"data: {len(train)} train / {len(val)} val examples, mean len {sum(len(e[0]) for e in train)/max(len(train),1):.0f}")
    steps_total = max(1, int(args.tokens // (args.tokens_per_batch * args.accum)))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    step, tokens_seen = 0, 0
    if args.resume and ckpt_dir.exists():
        step, tokens_seen = load_ckpt(model, ckpt_dir, device, opt)
        log(f"resumed at step {step}, {tokens_seen:,} tokens")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = min(1.0, (s - args.warmup) / max(1, steps_total - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    report = Reporter(run_dir, args.name)
    report({"event": "start", "steps_total": steps_total, "device": str(device), "counts": counts, "args": vars(args), "resumed_step": step})
    v = evaluate(model, val, device, pad, dtype)
    log(f"step {step} val {v:.4f} rss {rss_gb():.1f}G")
    report({"step": step, "val": v, "samples": sample(model, tok, device)})
    log(f"samples done rss {rss_gb():.1f}G")

    epoch = 0
    t_log, tok_log, loss_acc, loss_n = time.perf_counter(), 0, 0.0, 0
    micro = 0
    while step < steps_total:
        for ids, labels in batches(train, args.tokens_per_batch, pad, shuffle_seed=args.seed + epoch):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(input_ids=ids, attention_mask=(ids != pad)).logits
            n_tgt = int((labels[:, 1:] != -100).sum())
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / args.accum).backward()
            loss_acc += float(loss.detach())
            del logits, loss
            loss_n += 1
            tokens_seen += int(ids.numel())
            tok_log += int(ids.numel())
            micro += 1
            if micro % args.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = time.perf_counter() - t_log
                rec = {"step": step, "loss": loss_acc / loss_n, "lr": lr_at(step - 1), "grad_norm": float(gn), "tok_s": tok_log / dt, "tokens": tokens_seen, "steps_total": steps_total, "eta_s": (steps_total - step) * dt / args.log_every, "rss_gb": rss_gb()}
                log(f"step {step}/{steps_total} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['tok_s']:.0f} tok/s rss {rec['rss_gb']:.1f}G eta {rec['eta_s']/60:.0f}m")
                report(rec)
                t_log, tok_log, loss_acc, loss_n = time.perf_counter(), 0, 0.0, 0
            if step % args.eval_every == 0:
                free_cache(device)
                v = evaluate(model, val, device, pad, dtype)
                s = sample(model, tok, device)
                log(f"step {step} val {v:.4f} | sample: {s[0]['reply'][:160]!r} ({s[0]['tokens']} tok)")
                report({"step": step, "val": v, "samples": s})
            if step % args.ckpt_every == 0:
                save_ckpt(model, config, tok_path, ckpt_dir, step, tokens_seen, opt)
                log(f"ckpt saved at step {step}")
            if step >= steps_total:
                break
        epoch += 1
    save_ckpt(model, config, tok_path, ckpt_dir, step, tokens_seen, opt)
    v = evaluate(model, val, device, pad, dtype)
    s = sample(model, tok, device)
    report({"step": step, "val": v, "samples": s, "event": "done"})
    log(f"done: step {step} val {v:.4f}")
    export_int8(model, config, tok_path, base, run_dir / "export", log)
    if args.push:
        push_export(run_dir, args.push, log)


if __name__ == "__main__":
    main()
