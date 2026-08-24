"""Blind A/B rating between two checkpoints -- `babble ab run|rate|report|rollback`.

Held-out corpus loss (the promotion gate in `posttrain.py`) cannot see the
thing a human actually cares about: whether a candidate repeats the patterns
people corrected it toward, and whether it simply *reads* better. This module
does not gate promotion -- post-train ships on the loss gate exactly as it
does today -- it gives humans a cheap, reversible second opinion after the
fact:

1. `run` samples the same fixed prompt set from two checkpoints with
   identical decoding params and a shared per-prompt seed, so the only thing
   that can make the two responses differ is the weights. Which response
   lands on displayed side A vs B is an independent per-prompt coin flip; the
   true mapping is written to the session file but never shown to a rater.
2. `rate` walks a session in the terminal, blind, writing each vote back to
   disk immediately so an interrupted session resumes instead of restarting.
3. `report` unblinds a session, tallies the votes, and runs an exact
   binomial sign test (hand-rolled -- no scipy dependency here).
4. `rollback` restores the checkpoint a promotion archived in
   `posttrain._archive_outgoing_checkpoint` back into `latest.pt`, atomically,
   if the humans say the promotion made things worse.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from .blocklist import Blocklist
from .config import Settings
from .cpu_runtime import configure_cpu, force_cpu_device
from .generate import continue_text, sample, tokenizer_for_checkpoint
from .identity import Pseudonymiser
from .model import Babbler, ModelConfig
from .pairsplit import pair_split
from .post_state import record_rollback, trainable_pairs
from .trainer import SCRATCH_DIR
from .util import atomic_write_text, utcnow_iso, utcnow_stamp

#: A small fixed set of non-correction prompts, so a rating session is never
#: 100% corrections -- ro asked for "a few post trains" judged on how they
#: read generally, not only on whether they parroted a correction back.
EVERYDAY_PROMPTS = [
    "hello",
    "how are you",
    "what do you think about that",
    "tell me a story",
    "what's up",
    "good morning",
]

DEFAULT_PROMPT_COUNT = 20

#: Below this many decisive (a/b) votes, the sign test is honest but useless
#: -- report explicitly says so rather than printing a confident percentage.
MIN_DECISIVE_VOTES = 10
SIGNIFICANCE_LEVEL = 0.05

VALID_VOTES = ("a", "b", "tie", "skip")

__all__ = [
    "ABItem",
    "ABReport",
    "ABSession",
    "DEFAULT_PROMPT_COUNT",
    "EVERYDAY_PROMPTS",
    "MIN_DECISIVE_VOTES",
    "SIGNIFICANCE_LEVEL",
    "apply_vote",
    "binomial_sign_test",
    "build_report",
    "load_session",
    "rate_interactive",
    "render_report",
    "rollback",
    "run_ab",
    "save_session",
    "select_prompts",
    "unvoted_items",
]


# --- prompt selection ------------------------------------------------------


def select_prompts(held_out_prompts: list[str], count: int) -> list[tuple[str, str]]:
    """Prompts for one A/B session: the fixed everyday set plus held-out
    correction prompts, capped at `count` and de-duplicated by text.

    "Held-out" means not trained on in the run being evaluated -- callers
    pass the val-side correction prompts from `pairsplit.pair_split`, the
    exact split the promotion gate itself holds out, so the rating set can
    never overlap what either checkpoint's post-train actually saw.
    """
    everyday = [(p, "everyday") for p in EVERYDAY_PROMPTS]
    remaining = max(0, count - len(everyday))
    correction = [(p, "correction") for p in held_out_prompts[:remaining]]
    seen: set[str] = set()
    selected: list[tuple[str, str]] = []
    for prompt, source in everyday + correction:
        if prompt in seen:
            continue
        seen.add(prompt)
        selected.append((prompt, source))
    return selected[:count]


def _seed_for_prompt(seed: int, index: int, prompt: str) -> int:
    """A deterministic per-prompt seed, fed identically to both checkpoints'
    decode so the only thing that can make the two responses differ is the
    weights. sha256, not the builtin hash(): PYTHONHASHSEED randomises str
    hash() per process, which would desync the seed between the two model
    loads in the same `run_ab` call, and between separate runs of it."""
    digest = hashlib.sha256(f"{seed}:{index}:{prompt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


# --- checkpoint loading + sampling -----------------------------------------


def _load_checkpoint(path: Path) -> tuple[Babbler, int | None]:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig.from_dict(payload["config"])
    tok = tokenizer_for_checkpoint(path, cfg.vocab_size)
    model = Babbler(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.tokenizer = tok
    model.eval()
    return model, payload.get("step")


def _respond(model: Babbler, prompt: str, settings: Settings, seed: int) -> str:
    """One deterministic response, in whichever layout `settings.serve_layout`
    says the bot actually serves -- the same choice `CheckpointGenerator`
    makes, so an A/B session judges what a rater would actually see live."""
    generator = torch.Generator().manual_seed(seed)
    respond = sample if settings.serve_layout == "pair" else continue_text
    return respond(
        model,
        prompt,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        repetition_penalty=settings.repetition_penalty,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        generator=generator,
    )


# --- session data shape ------------------------------------------------


@dataclass
class ABItem:
    """One prompt's pair of responses.

    `response_candidate`/`response_baseline` and `a_is_candidate` are the
    true mapping -- persisted here so `report` can unblind, but never read by
    `rate_interactive`, which only ever looks at `response_a`/`response_b`.
    """

    index: int
    prompt: str
    prompt_source: str  # "everyday" | "correction"
    response_candidate: str
    response_baseline: str
    a_is_candidate: bool
    vote: str | None = None
    voted_at: str | None = None

    @property
    def response_a(self) -> str:
        return self.response_candidate if self.a_is_candidate else self.response_baseline

    @property
    def response_b(self) -> str:
        return self.response_baseline if self.a_is_candidate else self.response_candidate

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "response_candidate": self.response_candidate,
            "response_baseline": self.response_baseline,
            "a_is_candidate": self.a_is_candidate,
            "vote": self.vote,
            "voted_at": self.voted_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ABItem":
        return cls(
            index=int(raw["index"]),
            prompt=raw["prompt"],
            prompt_source=raw.get("prompt_source", "correction"),
            response_candidate=raw["response_candidate"],
            response_baseline=raw["response_baseline"],
            a_is_candidate=bool(raw["a_is_candidate"]),
            vote=raw.get("vote"),
            voted_at=raw.get("voted_at"),
        )


@dataclass
class ABSession:
    session_id: str
    created_at: str
    seed: int
    candidate_checkpoint: str
    baseline_checkpoint: str
    candidate_step: int | None
    baseline_step: int | None
    sampling: dict
    items: list[ABItem]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "candidate_checkpoint": self.candidate_checkpoint,
            "baseline_checkpoint": self.baseline_checkpoint,
            "candidate_step": self.candidate_step,
            "baseline_step": self.baseline_step,
            "sampling": self.sampling,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ABSession":
        return cls(
            session_id=raw["session_id"],
            created_at=raw["created_at"],
            seed=int(raw["seed"]),
            candidate_checkpoint=raw["candidate_checkpoint"],
            baseline_checkpoint=raw["baseline_checkpoint"],
            candidate_step=raw.get("candidate_step"),
            baseline_step=raw.get("baseline_step"),
            sampling=raw.get("sampling", {}),
            items=[ABItem.from_dict(d) for d in raw.get("items", [])],
        )


def load_session(path: Path) -> ABSession:
    return ABSession.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_session(session: ABSession, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(session.to_dict(), indent=2, ensure_ascii=False))


# --- run ---------------------------------------------------------------


def run_ab(
    settings: Settings,
    *,
    checkpoint_a: Path | None = None,
    checkpoint_b: Path | None = None,
    count: int | None = None,
    seed: int = 1,
    out_path: Path | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> tuple[ABSession, Path]:
    """Sample both checkpoints over the same fixed prompt set and write a
    blind rating session to disk.

    `checkpoint_a` ("the candidate") defaults to `latest.pt`, `checkpoint_b`
    ("the baseline") to `previous.pt` -- the predecessor a promotion archived
    before overwriting `latest.pt` (see `posttrain._archive_outgoing_checkpoint`).
    Sampling is deterministic and identical on both sides: the same
    per-prompt seed drives both checkpoints' decode, so the only thing that
    can make the two responses differ is the weights. Which side of the
    blind A/B a response lands on is a fresh per-prompt coin flip.
    """
    settings.ensure_dirs()
    checkpoint_a = Path(checkpoint_a) if checkpoint_a else settings.latest_checkpoint
    checkpoint_b = Path(checkpoint_b) if checkpoint_b else settings.previous_checkpoint
    if not checkpoint_a.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {checkpoint_a}")
    if not checkpoint_b.exists():
        raise FileNotFoundError(
            f"baseline checkpoint not found: {checkpoint_b} -- no archived predecessor yet "
            f"(a post-train promotion writes one); pass --b to compare against something else"
        )

    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    pairs = trainable_pairs(settings, ids, blocklist)
    _, val_pairs = pair_split(pairs)
    held_out_prompts = [p.prompt for p in val_pairs]
    prompts = select_prompts(held_out_prompts, count or DEFAULT_PROMPT_COUNT)

    configure_cpu(getattr(settings, "infer_threads", None) or settings.train_threads)
    model_a, step_a = _load_checkpoint(checkpoint_a)
    model_b, step_b = _load_checkpoint(checkpoint_b)

    coin = random.Random(seed)
    items: list[ABItem] = []
    for i, (prompt, source) in enumerate(prompts):
        gen_seed = _seed_for_prompt(seed, i, prompt)
        text_a = _respond(model_a, prompt, settings, gen_seed)
        text_b = _respond(model_b, prompt, settings, gen_seed)
        items.append(
            ABItem(
                index=i,
                prompt=prompt,
                prompt_source=source,
                response_candidate=text_a,
                response_baseline=text_b,
                a_is_candidate=coin.random() < 0.5,
            )
        )

    session_id = f"{utcnow_stamp()}-{secrets.token_hex(3)}"
    session = ABSession(
        session_id=session_id,
        created_at=utcnow_iso(),
        seed=seed,
        candidate_checkpoint=str(checkpoint_a),
        baseline_checkpoint=str(checkpoint_b),
        candidate_step=step_a,
        baseline_step=step_b,
        sampling={
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "repetition_penalty": settings.repetition_penalty,
            "frequency_penalty": settings.frequency_penalty,
            "presence_penalty": settings.presence_penalty,
            "no_repeat_ngram_size": settings.no_repeat_ngram_size,
            "serve_layout": settings.serve_layout,
        },
        items=items,
    )
    resolved_out = Path(out_path) if out_path else settings.ab_sessions_dir / f"ab-{session_id}.json"
    save_session(session, resolved_out)
    return session, resolved_out


# --- rate ----------------------------------------------------------------


def unvoted_items(session: ABSession) -> list[ABItem]:
    return [item for item in session.items if item.vote is None]


def apply_vote(session: ABSession, index: int, vote: str) -> None:
    if vote not in VALID_VOTES:
        raise ValueError(f"unknown vote {vote!r} -- expected one of {VALID_VOTES}")
    for item in session.items:
        if item.index == index:
            item.vote = vote
            item.voted_at = utcnow_iso()
            return
    raise KeyError(f"no item with index {index}")


def rate_interactive(session_path: Path, *, input_fn=input, print_fn=print) -> None:
    """Walk every unvoted item in the session, blind, prompting a keypress
    for A / B / tie / skip and writing the vote back to disk immediately.

    Only ever prints `prompt`, `response_a`, `response_b` -- never
    `a_is_candidate` or which checkpoint produced which text, so a rater has
    no way to learn "A is always the new one" across a session. Quitting
    mid-session (`q`) leaves every vote taken so far written to disk; the
    next `rate` call on the same file resumes at the first unvoted item
    instead of restarting.
    """
    session_path = Path(session_path)
    session = load_session(session_path)
    pending = unvoted_items(session)
    if not pending:
        print_fn("Nothing left to rate -- every item in this session already has a vote.")
        return
    print_fn(f"Rating {len(pending)} of {len(session.items)} prompt(s). a/b/t(ie)/s(kip), q to stop.")
    for item in pending:
        print_fn("")
        print_fn(f"prompt:     {item.prompt}")
        print_fn(f"response A: {item.response_a}")
        print_fn(f"response B: {item.response_b}")
        while True:
            raw = input_fn("[a/b/t/s/q] > ").strip().lower()
            if raw in ("a", "b"):
                apply_vote(session, item.index, raw)
                break
            if raw in ("t", "tie"):
                apply_vote(session, item.index, "tie")
                break
            if raw in ("s", "skip"):
                apply_vote(session, item.index, "skip")
                break
            if raw in ("q", "quit"):
                save_session(session, session_path)
                print_fn("Stopped -- resume any time with the same session file.")
                return
            print_fn("Not understood -- a, b, t(ie), s(kip), or q(uit).")
        save_session(session, session_path)  # written after every vote, not just at the end
    print_fn("Done -- every prompt in this session has a vote.")


# --- report ----------------------------------------------------------------


def binomial_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign-test p-value under a fair-coin null
    (p=0.5) -- `math.comb`, not scipy, which is not a project dependency.

    For a symmetric (p=0.5) binomial, the standard two-sided exact p-value
    reduces to `min(1, 2 * P(X <= min(wins, losses)))`: the smaller tail,
    doubled, capped at 1. `ties`/`skips` never enter this -- only decisive
    (a/b) votes are a trial.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, (2 * tail) / (2**n))


@dataclass
class ABReport:
    candidate_wins: int
    baseline_wins: int
    ties: int
    skips: int
    decisive: int
    candidate_win_rate: float | None
    p_value: float | None
    rows: list[dict]


def build_report(session: ABSession) -> ABReport:
    """Unblind every rated item and tally the votes.

    Unrated items (no vote yet) are silently excluded -- `report` on a
    partially-rated session tallies what has actually been rated so far.
    """
    candidate_wins = baseline_wins = ties = skips = 0
    rows: list[dict] = []
    for item in session.items:
        if item.vote is None:
            continue
        if item.vote == "tie":
            winner = "tie"
            ties += 1
        elif item.vote == "skip":
            winner = "skip"
            skips += 1
        elif item.vote == "a":
            winner = "candidate" if item.a_is_candidate else "baseline"
        elif item.vote == "b":
            winner = "baseline" if item.a_is_candidate else "candidate"
        else:
            continue  # an unrecognised vote value never contributes to the tally
        if winner == "candidate":
            candidate_wins += 1
        elif winner == "baseline":
            baseline_wins += 1
        rows.append(
            {
                "index": item.index,
                "prompt": item.prompt,
                "prompt_source": item.prompt_source,
                "candidate_response": item.response_candidate,
                "baseline_response": item.response_baseline,
                "vote": item.vote,
                "winner": winner,
            }
        )
    decisive = candidate_wins + baseline_wins
    return ABReport(
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        skips=skips,
        decisive=decisive,
        candidate_win_rate=(candidate_wins / decisive) if decisive else None,
        p_value=binomial_sign_test(candidate_wins, baseline_wins) if decisive else None,
        rows=rows,
    )


def render_report(session: ABSession, report: ABReport) -> str:
    lines = [
        f"session:   {session.session_id}",
        f"candidate: {session.candidate_checkpoint} (step {session.candidate_step})",
        f"baseline:  {session.baseline_checkpoint} (step {session.baseline_step})",
        f"seed:      {session.seed}",
        "",
    ]
    for row in report.rows:
        lines.append(f"[{row['index']:>3}] ({row['prompt_source']}) {row['prompt']!r}")
        lines.append(f"      candidate: {row['candidate_response']!r}")
        lines.append(f"      baseline:  {row['baseline_response']!r}")
        lines.append(f"      vote: {row['vote']} -> {row['winner']}")
    lines.append("")
    rated = report.candidate_wins + report.baseline_wins + report.ties + report.skips
    lines.append(
        f"{rated}/{len(session.items)} rated -- candidate {report.candidate_wins}, "
        f"baseline {report.baseline_wins}, ties {report.ties}, skipped {report.skips}"
    )
    if report.decisive == 0:
        lines.append("No decisive (a/b) votes yet -- nothing to conclude.")
        return "\n".join(lines)

    lines.append(
        f"candidate win rate (of decisive votes): {report.candidate_win_rate:.0%} "
        f"({report.candidate_wins}/{report.decisive})"
    )
    lines.append(f"two-sided sign-test p-value: {report.p_value:.4f}")
    if report.decisive < MIN_DECISIVE_VOTES:
        lines.append(
            f"Only {report.decisive} decisive vote(s) -- this sample is too small to conclude "
            f"anything either way, whatever the percentage above looks like."
        )
    elif report.p_value >= SIGNIFICANCE_LEVEL:
        lines.append(
            "Not statistically significant (p >= 0.05) -- could easily be noise, not a real "
            "difference between the two checkpoints."
        )
    else:
        winner = "candidate" if report.candidate_wins > report.baseline_wins else "baseline"
        lines.append(f"Statistically significant (p < 0.05): raters preferred the {winner}.")
    return "\n".join(lines)


# --- rollback ----------------------------------------------------------------


def rollback(settings: Settings) -> dict:
    """Restore the archived predecessor (`previous.pt`) into `latest.pt`,
    atomically, and record it so the next post-train's `pretrained.pt`
    snapshot discipline treats the restored weights correctly (see
    `post_state.record_rollback`)."""
    previous = settings.previous_checkpoint
    if not previous.exists():
        raise FileNotFoundError(
            f"no archived predecessor at {previous} -- nothing to roll back to yet "
            f"(a post-train promotion archives one there)"
        )
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    scratch.mkdir(exist_ok=True)
    tmp = scratch / "latest.pt.rollback.tmp"
    shutil.copyfile(previous, tmp)
    os.replace(tmp, settings.latest_checkpoint)

    restored_step = None
    try:
        payload = torch.load(previous, map_location="cpu", weights_only=True)
        restored_step = payload.get("step")
    except Exception:
        pass

    record_rollback(settings, restored_from=str(previous), restored_step=restored_step)
    return {"restored_from": str(previous), "restored_step": restored_step}
