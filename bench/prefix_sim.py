"""Replay multi-turn conversations through the real bot path and measure TTFT.

    PYTHONPATH=<tree> python bench/prefix_sim.py --label main --out /tmp/x.json

Drives `core.Babble.handle_message` exactly the way `bot.py` does (reply
chains via `remember`, live conversation settings), backed by the native
generator on the live model directory (read only). Every generation's prompt
length, reused prefix tokens and TTFT are recorded by wrapping the
generator's `_generate`. If the tree under test has `Babble.prewarm` (the
background pre-prefill), it is run after each reply in a separate thread,
just like the bot does, during the simulated think time.

Works against main's tree too (no prewarm, no overflow setting), so the
baseline is measured with main's code, not with flags flipped in this one.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

MODEL_DIR = "/home/jason/projects/babble/artifacts/hf-booper-longctx-v1"

USER_MESSAGES = [
    "hey booper",
    "what are you up to today",
    "lol same. i'm supposed to be studying for my chem exam but i keep procrastinating",
    "it's organic chem. reaction mechanisms are killing me",
    "do you know what an sn2 reaction is",
    "ok but like explain it like i'm five",
    "that kinda makes sense. what about sn1",
    "why does the carbocation matter so much",
    "my professor talks so fast i can't keep up with any of it",
    "anyway enough school. what music are you into",
    "have you ever heard of radiohead? ok computer is my favorite album ever",
    "paranoid android is like 6 minutes of pure chaos and i love it",
    "what's the best song to study to in your opinion",
    "lofi is fine but it makes me sleepy honestly",
    "speaking of sleep i got like 4 hours last night",
    "because i was up playing elden ring until 3am like an idiot",
    "the boss i'm stuck on is malenia. she's impossible",
    "waterfowl dance destroys me every single time",
    "do you think i should just summon help or is that cheating",
    "ok fine i'll summon mimic tear. no shame",
    "wait what were we talking about before games",
    "oh right chem. ugh",
    "can you quiz me on something easy",
    "what's the difference between an alkane and an alkene",
    "nice i actually knew that one",
    "ok one more. what's a nucleophile",
    "you're a better teacher than my professor tbh",
    "i should probably go actually study now",
    "thanks for keeping me company booper",
    "goodnight!",
]


#: For --long: people paste things. Each message carries ~120-180 tokens of
#: this, so eight turns overflow the 1536-token budget before the turn cap.
PASTE = (
    "ok so here is the paragraph from my notes that i am trying to understand, sorry it is long: "
    "in a bimolecular nucleophilic substitution the nucleophile attacks the electrophilic carbon from "
    "the side opposite the leaving group, so bond formation and bond breaking happen in a single "
    "concerted step through a trigonal bipyramidal transition state. the rate depends on the "
    "concentration of both the substrate and the nucleophile, and steric hindrance around the carbon "
    "slows the reaction dramatically, which is why methyl and primary substrates react fastest while "
    "tertiary substrates barely react at all. the stereochemistry at the carbon is inverted, like an "
    "umbrella flipping inside out in the wind. polar aprotic solvents such as acetone, dmso and dmf "
    "speed it up because they do not cage the nucleophile in a shell of hydrogen bonds. in contrast "
    "the unimolecular pathway goes through a carbocation intermediate, so its rate only depends on "
    "the substrate, it favors tertiary carbons, and it scrambles stereochemistry into a racemic mix."
).split(" ")


def long_message(turn: int, text: str) -> str:
    n = 70 + (turn * 37) % 60
    start = (turn * 23) % len(PASTE)
    words = (PASTE[start:] + PASTE)[:n]
    return f"{text}\n> {' '.join(words)}"


def build(args):
    from babble.config import Settings
    from babble.core import Babble
    from babble.identity import Pseudonymiser
    from babble.logs import EventLog
    from babble.hfserve import make_generator

    root = Path(args.root)
    s = Settings.for_root(root)
    s.salt = "prefix-sim-salt-not-real"
    s.serve_backend = "hf"
    s.hf_runtime = "native"
    s.hf_model_dir = Path(MODEL_DIR)
    s.serve_layout = "pair"
    s.conversation_context = True
    s.conversation_max_turns = args.max_turns
    s.conversation_max_tokens = args.max_tokens
    s.conversation_max_chars = 0
    s.max_new_tokens = args.max_new_tokens
    s.best_of = 4
    s.temperature, s.top_k, s.top_p = 0.5, 40, 0.9
    s.repetition_penalty, s.no_repeat_ngram_size = 1.15, 4
    s.lean_prefix_cache_mb = args.cache_mb
    s.train_trigger_rows = 0
    s.post_trigger_pairs = 0
    for key, value in args.set or []:
        cur = getattr(s, key)
        setattr(s, key, type(cur)(value) if cur is not None else value)
    s.ensure_dirs()
    log = EventLog(s, Pseudonymiser.load(s), component="prefix-sim")
    gen = make_generator(s, log)
    brain = Babble(s, generator=gen, log=log, bot_user_id="bot-9999")
    return s, gen, brain


class Recorder:
    def __init__(self, gen):
        self.gen = gen
        self.calls = []
        inner = gen._generate

        def wrapped(prompt, **kw):
            text, result = inner(prompt, **kw)
            P = len(gen._encode_prompt(prompt)[0])
            self.calls.append(
                dict(
                    prompt=prompt,
                    prompt_tokens=P,
                    reused=int(result.prefix_reused),
                    ttft_ms=result.stats.ttft_s * 1000,
                    total_ms=result.stats.total_s * 1000,
                    reply=text,
                )
            )
            return text, result

        gen._generate = wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/tmp/maxperf/prefix/simroot")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--think-s", type=float, default=2.0)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=509)
    ap.add_argument("--cache-mb", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-prewarm", action="store_true")
    ap.add_argument("--long", action="store_true", help="paste-heavy messages that overflow the token budget")
    ap.add_argument("--set", nargs=2, action="append", metavar=("FIELD", "VALUE"))
    args = ap.parse_args()

    import shutil

    shutil.rmtree(args.root, ignore_errors=True)
    from babble.core import IncomingMessage

    torch.manual_seed(args.seed)
    s, gen, brain = build(args)
    rec = Recorder(gen)
    prewarm = None if args.no_prewarm else getattr(brain, "prewarm", None)

    ids = iter(range(10_000, 10**9))
    users = [f"{900000000000000000 + c}" for c in range(args.channels)]

    def dispatch(msg):
        posted = []
        for reply in brain.handle_message(msg):
            mid = str(next(ids))
            brain.remember(mid, reply)
            posted.append((mid, reply))
        return posted

    def incoming(c, text, reply_to=None):
        return IncomingMessage(
            message_id=str(next(ids)), author_id=users[c], content=text, channel_id=f"chan-{c}",
            guild_id="guild-1", mentions_bot=reply_to is None,
            reply_to_message_id=reply_to, reply_to_is_bot=reply_to is not None,
        )

    for c in range(args.channels):  # consent, exactly as a real user gets past it
        dispatch(incoming(c, "hello"))
        dispatch(IncomingMessage(message_id=str(next(ids)), author_id=users[c], content="!babble accept",
                                 channel_id=f"chan-{c}", guild_id="guild-1"))
    rec.calls.clear()

    last = [None] * args.channels
    rows = []
    warm_threads: list[threading.Thread] = []
    for turn in range(args.turns):
        for c in range(args.channels):
            text = USER_MESSAGES[turn % len(USER_MESSAGES)]
            if args.long and turn % 2 == 1:
                text = long_message(turn, text)
            if args.channels > 1:
                text = f"{text}" if c == 0 else f"{text} ({c})"
            n_before = len(rec.calls)
            t0 = time.perf_counter()
            posted = dispatch(incoming(c, text, last[c]))
            wall = (time.perf_counter() - t0) * 1000
            gen_rows = rec.calls[n_before:]
            if not gen_rows:
                raise SystemExit(f"turn {turn} chan {c}: no generation ({[r.content for _, r in posted]})")
            g = gen_rows[-1]
            mid, reply = next((m, r) for m, r in posted if r.kind == "generation")
            last[c] = mid
            row = dict(turn=turn + 1, channel=c, prompt_tokens=g["prompt_tokens"], reused=g["reused"],
                       ttft_ms=round(g["ttft_ms"], 2), total_ms=round(g["total_ms"], 1),
                       core_overhead_ms=round(wall - g["total_ms"], 2),
                       visible_turns=g["prompt"].count("\nassistant: "), user=text, reply=reply.content,
                       prompt=g["prompt"])
            rows.append(row)
            print(f"t{turn + 1:02d} c{c} P={row['prompt_tokens']:5d} reused={row['reused']:5d} "
                  f"ttft={row['ttft_ms']:7.1f}ms turns={row['visible_turns']} core+={row['core_overhead_ms']:.1f}ms",
                  flush=True)
            if prewarm is not None:
                th = threading.Thread(target=prewarm, args=(reply,), daemon=True)
                th.start()
                warm_threads.append(th)
        time.sleep(args.think_s)
    for th in warm_threads:
        th.join()

    ttfts = [r["ttft_ms"] for r in rows]
    q = statistics.quantiles(ttfts, n=10)
    summary = dict(label=args.label, n=len(ttfts), median_ms=statistics.median(ttfts), p90_ms=q[8],
                   mean_ms=statistics.fmean(ttfts), max_ms=max(ttfts),
                   cache=gen.prefix_cache.stats(), cache_mb=args.cache_mb, channels=args.channels)
    if hasattr(gen, "prewarm_stats"):
        summary["prewarm"] = gen.prewarm_stats()
    print(json.dumps(summary, indent=1))
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=rows, args=vars(args)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
