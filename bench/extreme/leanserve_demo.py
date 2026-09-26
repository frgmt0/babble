"""Real-model gates for the lean runtime that are not in pytest.

Run under the shared lock with the serving env of `babble bench`
(BABBLE_SERVE_BACKEND=hf, BABBLE_HF_MODEL_DIR=..., BABBLE_CONVERSATION_CONTEXT=1,
BABBLE_NO_REPEAT_NGRAM_SIZE=4), choosing the runtime with BABBLE_HF_RUNTIME:

    python bench/extreme/leanserve_demo.py rss        # RSS after load + one reply
    python bench/extreme/leanserve_demo.py ttft       # multi-turn TTFT per turn
    python bench/extreme/leanserve_demo.py samples    # 10 seeded replies

Every prompt here is synthetic; nothing reads the corpus.
"""

from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from babble.config import Settings  # noqa: E402
from babble.conversation import ConversationTurn  # noqa: E402
from babble.hfserve import make_generator  # noqa: E402

SHORT_PROMPTS = [
    "hey booper whats up",
    "do you like cats or dogs",
    "what should i eat for dinner",
    "lol that game last night was wild",
    "i cant sleep",
    "whats your favorite song rn",
    "booper say something funny",
    "im so bored in class",
    "did you finish your homework",
    "good morning!!",
]

CONVERSATION = [
    "hey booper, i just got home from work and im so tired, today was a really long day",
    "yeah my boss made us redo the whole presentation because the client changed their mind again",
    "honestly i think im gonna order food tonight, i dont have the energy to cook anything",
    "thinking about pizza or maybe thai food, the place down the street has really good pad thai",
    "ok pad thai it is. what are you up to tonight, any plans or just chilling",
    "nice. i might watch a movie after dinner, do you have any recommendations for something chill",
    "i have seen that one already, something newer maybe, it came out like last year",
]


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _prompt(gen, settings, history, user):
    return gen.conversation_prompt(
        history,
        user,
        max_turns=settings.conversation_max_turns,
        max_tokens=settings.conversation_max_tokens,
        max_chars=settings.conversation_max_chars,
    )


def rss(settings: Settings) -> dict:
    base = _rss_mb()
    t0 = time.perf_counter()
    gen = make_generator(settings)
    load_s = time.perf_counter() - t0
    after_load = _rss_mb()
    torch.manual_seed(0)
    gen(_prompt(gen, settings, [], "hey booper whats up"))
    return {
        "runtime": settings.hf_runtime,
        "prefill_fp32": getattr(settings, "lean_prefill_fp32", None),
        "baseline_mb": round(base),
        "after_load_mb": round(after_load),
        "after_reply_mb": round(_rss_mb()),
        "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "load_s": round(load_s, 2),
    }


def _ttft(gen, prompt: str, *, cached: bool, best_of: int, max_new: int):
    """(ttft_ms, reply, prefix_tokens_reused) for one generation."""
    if hasattr(gen, "prefix_cache"):
        text, result = gen._generate(prompt, max_new_tokens=max_new, best_of=best_of, use_prefix_cache=cached)
        return result.stats.ttft_s * 1000, text, result.prefix_reused
    text, stats = gen._generate(prompt, max_new_tokens=max_new, best_of=best_of)
    return stats.ttft_s * 1000, text, 0


def ttft(settings: Settings, repeats: int = 3) -> list[dict]:
    """Walk one growing conversation. Per turn: cold TTFT (no prefix reuse) and,
    for lean, warm TTFT with the prefix cache primed by the previous turn."""
    gen = make_generator(settings)
    best_of = settings.best_of
    lean = hasattr(gen, "prefix_cache")
    history: list[ConversationTurn] = []
    rows = []
    torch.manual_seed(0)
    _ttft(gen, _prompt(gen, settings, [], "warm up"), cached=False, best_of=best_of, max_new=2)
    for turn, user in enumerate(CONVERSATION):
        prompt = _prompt(gen, settings, history, user)
        n_tok = int(gen._encode_prompt(prompt).shape[-1])
        cold = [_ttft(gen, prompt, cached=False, best_of=best_of, max_new=1)[0] for _ in range(repeats)]
        row = {"turn": turn + 1, "prompt_tokens": n_tok, "cold_ttft_ms": round(statistics.median(cold), 1)}
        if lean:
            # Re-prime with the previous turn's prompt each repeat so every warm
            # measurement is a genuine first hit, then keep this turn's entry.
            warm = []
            for _ in range(repeats):
                if history:
                    prev = _prompt(gen, settings, history[:-1], history[-1].user)
                    gen.prefix_cache.clear()
                    _ttft(gen, prev, cached=True, best_of=1, max_new=1)
                ms, _, reused = _ttft(gen, prompt, cached=True, best_of=best_of, max_new=1)
                warm.append(ms)
            row.update(warm_ttft_ms=round(statistics.median(warm), 1), reused_tokens=reused)
        # The reply that becomes history (a real sampled reply, capped short).
        torch.manual_seed(turn)
        _, reply, _ = _ttft(gen, prompt, cached=True, best_of=best_of, max_new=24)
        history.append(ConversationTurn(user, reply or "ok"))
        rows.append(row)
    return rows


def samples(settings: Settings) -> list[dict]:
    gen = make_generator(settings)
    out = []
    for i, text in enumerate(SHORT_PROMPTS):
        torch.manual_seed(1000 + i)
        reply = gen(_prompt(gen, settings, [], text))
        out.append({"prompt": text, "reply": reply.text, "ms": round(reply.ms)})
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "rss"
    settings = Settings.from_env()
    result = {"rss": rss, "ttft": ttft, "samples": samples}[mode](settings)
    if isinstance(result, list):
        for row in result:
            print(json.dumps({"runtime": settings.hf_runtime, **row}))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
