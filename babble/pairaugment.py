"""Correction-pair augmentation: LLM-paraphrased variants of real correction
pairs, so post-train has more than a few dozen `(prompt, chosen)` examples to
learn from without inventing an answer nobody actually corrected the bot
with.

ro's ask: "can we generate synthetic pairs? so like when i correct it we
create 3 possible variants that are semantically correct and then 50 turns
into 150?" -- for each real correction pair, paraphrase it into a handful of
variants that stay semantically tied to the original: the *response* must
still answer the (possibly reworded) prompt the same way. `("what cake is
best?" -> "chocolate")` may become `("whats the best cake" -> "chocolate,
obviously")`; it must never become `(... -> "vanilla")`.

**Why this cannot be `synthcorpus.py`'s markov chain.** That generator only
ever recombines observed corpus n-grams; it has no notion of "the same
meaning" and would as happily splice together a *wrong* answer as a right
one. Paraphrasing that preserves meaning needs a model that actually reads
the pair, which is what `babble/llm.py`'s `LLMClient` is for. This module is
never the corpus generator's replacement -- it is a second, additive path
that touches correction pairs, not corpus rows.

**The variants must sound like the person, not like an assistant.** A
paraphrase that is semantically correct but reads as polished, capitalised,
"Certainly! Here's..." assistant prose is *still a failure*: it would teach
the model a distribution of *text* it is not supposed to imitate, even
though every individual pair "passes" a meaning check. Every paraphrase call
is anchored with real, verbatim examples of the register actually being
imitated (`_style_examples`, drawn from the same person's other train-side
pairs) and instructed explicitly to preserve that register -- lowercase,
short, discord-shaped, no assistant framing. Whether that instruction
actually held is not something to assume: `register_comparison` below
measures it after the fact (length, casing, punctuation, vocabulary overlap
against the real pairs), and `babble augment-pairs` always prints it. See the
project report for what it found on this run.

**Train-only, by construction.** A variant may only be derived from a
correction pair on the *train* side of `pairsplit.pair_split` -- the same
split `posttrain.py` uses to hold out real pairs for validation. Generating
from a val-side pair would let paraphrased-but-recognisable val phrasing back
into training, exactly the leak `PIPELINE_REVAMP_2026-08-20.md` §7.1
documents for the corpus-level generator (364/1200 rows carrying val-only
trigrams, a fake 0.33 nat "improvement"). `generate_augmented_pairs` only
ever reads train-side pairs; `check_leakage` / `assert_no_leakage` below are
the automated, re-runnable proof that nothing stored violates that, and
`trainable_augmented_pairs` re-checks train-side membership (and consent, and
the blocklist) again at train time -- the same belt-and-braces pattern
`synthetic.py` and `synthcorpus.py` already use.

**Never silently garbage.** A model response that doesn't parse as the
requested JSON, doesn't have the requested shape, or comes back with an
empty half is not "close enough" -- `_parse_variants` raises
`ParaphraseError` rather than storing whatever text came back. Both that and
a broken call for one specific pair (`llm.LLMError` -- a timeout, a rate
limit, a transient CLI hiccup) only cost their own source pair: caught in
`generate_augmented_pairs`, counted in `AugmentGenerateResult.failed_pairs`,
and named in `.failures` -- never silently dropped from the report, and
never storing anything for that pair. A client that is broken for every pair
still surfaces loudly (every pair lands in `.failures`), it just does not
take the whole batch down on the first flaky call -- which is what makes
`AutoAugmentTrigger`'s single-pair call safe to fire-and-forget: a failure
there is one skipped pair in a subprocess nobody is watching, not a crash.

**Kept strictly apart from human corrections and from `synthetic.py`'s
pairs.** Its own file (`data/augmented_pairs.jsonl`,
`Settings.augmented_pairs_path`), every row carries `"synthetic": true`, and
`babble post-train` only trains on it when told to
(`--augment-pairs`/`Settings.post_augment_pairs`).

**A privacy note, stated once here rather than left implicit.** Unlike the
corpus-internal generators, paraphrasing genuinely sends the pair's text (and
a few other train-side pairs, as style anchors) to wherever `LLMClient`
routes it -- the `claude` CLI's own API call, off this box. That is a
material difference from every other generator in this project and is why
this stays an explicit, separate opt-in rather than an always-on step.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .blocklist import Blocklist
from .config import Settings
from .identity import Pseudonymiser
from .llm import LLMClient, LLMError, client_from_settings
from .logs import EventLog, NullLog
from .pairsplit import pair_split
from .post_state import trainable_pairs
from .store import Interaction
from .util import utcnow_iso

__all__ = [
    "AugmentedPair",
    "AugmentedPairStore",
    "AugmentGenerateResult",
    "ParaphraseError",
    "LeakageError",
    "LeakageReport",
    "RegisterReport",
    "AutoAugmentTrigger",
    "generate_augmented_pairs",
    "trainable_augmented_pairs",
    "augmented_pair_count",
    "check_leakage",
    "assert_no_leakage",
    "register_comparison",
]

METHOD_LLM_PARAPHRASE = "llm_paraphrase"

#: Cap on how much text a single variant half may run to before it is treated
#: as a malformed response rather than a paraphrase -- generous for this
#: corpus's short rows, tight enough to catch a model that ignored the ask
#: and wrote an essay.
_MAX_VARIANT_CHARS = 600


@dataclass(frozen=True)
class AugmentedPair:
    """One LLM-paraphrased variant of a real correction pair.

    `source_pair_id` is the `Interaction.id` it was derived from --
    provenance that both the leakage check and `trainable_augmented_pairs`
    depend on. `variant_index` numbers the variants requested for one source
    pair in one generation call (0-based), which is what lets an N-sweep
    (`babble/pairaugment.py`'s callers) select "the first k variants per
    pair" without regenerating anything.
    """

    id: str
    prompt: str
    chosen: str
    source_pair_id: str
    variant_index: int
    method: str = METHOD_LLM_PARAPHRASE
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "source_pair_id": self.source_pair_id,
            "variant_index": self.variant_index,
            "method": self.method,
            "synthetic": True,  # belt-and-braces: true even for a reader that skips the filename
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "AugmentedPair":
        return cls(
            id=raw["id"],
            prompt=raw.get("prompt", ""),
            chosen=raw.get("chosen", ""),
            source_pair_id=raw.get("source_pair_id", ""),
            variant_index=int(raw.get("variant_index", 0)),
            method=raw.get("method", METHOD_LLM_PARAPHRASE),
            created_at=raw.get("created_at", ""),
        )


def make_augmented_id(source_pair_id: str, variant_index: int, prompt: str, chosen: str) -> str:
    """Content-addressed over the source pair, which variant slot this is,
    and the actual generated text -- so a rerun that gets a different
    paraphrase from the model (LLM output is not deterministic the way the
    markov chain is) is stored as a fresh row rather than colliding with, or
    silently overwriting, the previous attempt."""
    payload = "\x1f".join(["augmented", source_pair_id, str(variant_index), prompt, chosen])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class AugmentedPairStore:
    """Append-only JSONL at `Settings.augmented_pairs_path`. Same
    append/dedupe contract as `SyntheticPairStore`, deliberately not shared
    code with it: a separate, small class is part of what keeps "never
    silently mixed into human corrections or into the postulated-prompt
    pairs" true rather than a convention someone has to remember."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: AugmentedPair) -> bool:
        return self.extend([row]) == 1

    def extend(self, rows: Iterable[AugmentedPair]) -> int:
        seen = self.ids()
        fresh = []
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            fresh.append(row)
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for row in fresh:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return len(fresh)

    def all(self) -> list[AugmentedPair]:
        if not self.path.exists():
            return []
        rows: list[AugmentedPair] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(AugmentedPair.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a torn line never takes the store down with it
        return rows

    def ids(self) -> set[str]:
        return {r.id for r in self.all()}

    def count(self) -> int:
        return len(self.all())


# --- consent-gated pair reads, mirroring synthetic.py's discipline --------


def _real_train_val_pairs(
    settings: Settings, ids: Pseudonymiser | None, blocklist: Blocklist | None
) -> tuple[list[Interaction], list[Interaction]]:
    """The real, currently-consented correction pairs, split train/val by
    `pairsplit.pair_split` -- the ONE split definition generation and
    training both use (see the module docstring)."""
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    pairs = trainable_pairs(settings, ids, blocklist)
    return pair_split(pairs)


# --- style anchoring: keep the paraphraser in the speaker's register ------


def _style_examples(train_pairs: list[Interaction], around: Interaction, limit: int = 6) -> list[str]:
    """A handful of *other* train-side pairs' real `chosen` text, used as
    verbatim register anchors in the paraphrase prompt -- see the module
    docstring's "must sound like the person, not an assistant" section.
    Deterministic per source pair (seeded on its id) so a rerun asks for the
    same anchors rather than a different sample each time."""
    pool = [p.chosen for p in train_pairs if p.id != around.id and p.chosen.strip()]
    if not pool:
        return [around.chosen] if around.chosen.strip() else []
    rng = random.Random(around.id)
    return rng.sample(pool, k=min(limit, len(pool)))


def _build_prompt(prompt: str, chosen: str, n: int, style_examples: list[str]) -> str:
    anchors = "\n".join(f"- {text}" for text in style_examples) or "- (no other examples available)"
    return (
        "You are generating training-data variants for a tiny language model "
        "being taught to imitate ONE specific person's real Discord chat voice.\n\n"
        "Verbatim examples of how this person actually writes -- lowercase, short, "
        "casual, discord-shaped, typos and slang left exactly as typed:\n"
        f"{anchors}\n\n"
        "Paraphrase the correction pair below. Preserve its meaning exactly -- do "
        "not change facts, opinions, or the substance of the answer -- but reword "
        "the wording. The rewritten response MUST stay in the voice shown above: "
        "lowercase (no sentence-case, no capitalizing the first word), short, no "
        "assistant-style politeness or framing (never 'Certainly!', 'I'd be happy "
        "to help', 'Here is...'), minimal punctuation, contractions and casual "
        "spelling where it reads naturally. A grammatically polished, "
        "assistant-sounding rewrite is a FAILURE even when the facts are correct.\n\n"
        f'original prompt: "{prompt}"\n'
        f'original response: "{chosen}"\n\n'
        f"Reply with EXACTLY {n} distinct variants as a raw JSON array -- no prose, "
        "no markdown code fences, nothing before or after the array. Each element: "
        '{"prompt": "...", "chosen": "..."}. "chosen" is required and must '
        'preserve the original response\'s meaning. "prompt" may be reworded too, '
        "as long as the response would still plausibly answer it -- if you can't "
        "reword it naturally, repeat the original prompt unchanged."
    )


class ParaphraseError(Exception):
    """A model response could not be trusted as a real variant -- malformed
    JSON, the wrong shape, an empty half, or an implausibly long response.
    Raised per source pair; never silently downgraded into a stored row."""


def _parse_variants(raw: str, prompt: str, chosen: str, n: int) -> list[tuple[str, str]]:
    text = raw.strip()
    # Models asked for "raw JSON, no fences" sometimes fence anyway -- strip
    # the fence rather than fail on it, but nothing past that is forgiven.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParaphraseError(f"response was not valid JSON: {text[:200]!r}") from exc
    if not isinstance(payload, list) or not payload:
        raise ParaphraseError(f"expected a non-empty JSON array, got: {text[:200]!r}")

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ParaphraseError(f"expected an object per variant, got: {item!r}")
        vp = str(item.get("prompt", "")).strip()
        vc = str(item.get("chosen", "")).strip()
        if not vp or not vc:
            raise ParaphraseError(f"variant had an empty half: {item!r}")
        if len(vp) > _MAX_VARIANT_CHARS or len(vc) > _MAX_VARIANT_CHARS:
            raise ParaphraseError(f"variant exceeded {_MAX_VARIANT_CHARS} chars: {item!r}")
        if vc == chosen and vp == prompt:
            continue  # not a paraphrase, just an echo -- worth less than nothing
        key = (vp, vc)
        if key in seen:
            continue
        seen.add(key)
        out.append((vp, vc))

    if not out:
        raise ParaphraseError("every variant was empty, an echo, or a duplicate")
    return out


def paraphrase_pair(
    client: LLMClient, prompt: str, chosen: str, n: int, style_examples: list[str]
) -> list[tuple[str, str]]:
    """Ask `client` for up to `n` semantically-equivalent variants of
    `(prompt, chosen)`, in the register `style_examples` anchors. Raises
    `ParaphraseError` for a response that cannot be trusted, or lets
    `LLMError` (the call itself failed) propagate -- `generate_augmented_
    pairs` catches both per pair, so either one only costs its own pair."""
    raw = client.complete(_build_prompt(prompt, chosen, n, style_examples))
    return _parse_variants(raw, prompt, chosen, n)


@dataclass
class AugmentGenerateResult:
    source_pairs: int = 0
    train_side_pairs: int = 0
    val_side_pairs: int = 0
    requested_per_pair: int = 0
    generated: int = 0
    skipped_duplicate: int = 0
    skipped_blocklist: int = 0
    skipped_already_covered: int = 0
    failed_pairs: int = 0
    failures: list[str] = field(default_factory=list)


def generate_augmented_pairs(
    settings: Settings,
    *,
    n: int = 3,
    client: LLMClient | None = None,
    max_workers: int = 1,
    pair_ids: Iterable[str] | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> AugmentGenerateResult:
    """Paraphrase train-side real correction pairs into up to `n` variants
    each and append the new ones to `settings.augmented_pairs_path`.

    Only ever reads train-side pairs (`pairsplit.pair_split`) -- val-side
    pairs are counted in the result (`val_side_pairs`) so the split is
    visible, but never handed to the paraphraser. `pair_ids`, when given,
    restricts generation to that subset of (still train-side-checked) pairs
    -- what `AutoAugmentTrigger` uses to paraphrase exactly the one pair that
    was just corrected, instead of re-sweeping everything. A pair that
    already has `n` or more stored variants is skipped
    (`skipped_already_covered`), which is what makes both the targeted call
    and a full rerun of `babble augment-pairs` cheap and safe to repeat.
    `max_workers` > 1 runs the (independent, stateless) per-pair calls
    concurrently, which matters once there are dozens of pairs and each call
    is a few seconds of CLI round-trip; sequential (the default) is what a
    test's fake client needs for deterministic ordering.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    client = client or client_from_settings(settings)

    pairs = trainable_pairs(settings, ids, blocklist)
    train_pairs, val_pairs = pair_split(pairs)
    if pair_ids is not None:
        wanted = set(pair_ids)
        train_pairs = [p for p in train_pairs if p.id in wanted]

    store = AugmentedPairStore(settings.augmented_pairs_path)
    existing = store.ids()
    covered: dict[str, int] = {}
    for row in store.all():
        covered[row.source_pair_id] = covered.get(row.source_pair_id, 0) + 1

    result = AugmentGenerateResult(
        source_pairs=len(pairs),
        train_side_pairs=len(train_pairs),
        val_side_pairs=len(val_pairs),
        requested_per_pair=n,
    )

    pending = [p for p in train_pairs if covered.get(p.id, 0) < n]
    result.skipped_already_covered = len(train_pairs) - len(pending)

    def work(pair: Interaction) -> tuple[Interaction, list[tuple[str, str]] | Exception]:
        try:
            variants = paraphrase_pair(
                client, pair.prompt, pair.chosen, n, _style_examples(train_pairs, pair)
            )
            return pair, variants
        except (ParaphraseError, LLMError) as exc:
            # Both a bad response (ParaphraseError) and a broken call for
            # THIS pair (LLMError -- timeout, rate limit, a transient CLI
            # hiccup) only cost this one pair's variants. A client that is
            # broken for every pair still surfaces loudly: every pair ends up
            # in `failures`, `failed_pairs` equals the whole batch, and
            # nothing is silently stored -- but one flaky call never takes
            # forty-nine good ones down with it.
            return pair, exc

    if max_workers > 1 and pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(work, pending))
    else:
        outcomes = [work(p) for p in pending]

    fresh: list[AugmentedPair] = []
    for pair, outcome in outcomes:
        if isinstance(outcome, Exception):
            result.failed_pairs += 1
            result.failures.append(f"{pair.id}: {outcome}")
            continue
        for idx, (vp, vc) in enumerate(outcome):
            if blocklist.matches(vp, vc):
                result.skipped_blocklist += 1
                continue
            aug_id = make_augmented_id(pair.id, idx, vp, vc)
            if aug_id in existing:
                result.skipped_duplicate += 1
                continue
            existing.add(aug_id)
            fresh.append(
                AugmentedPair(
                    id=aug_id, prompt=vp, chosen=vc, source_pair_id=pair.id, variant_index=idx
                )
            )

    result.generated = store.extend(fresh)
    return result


def trainable_augmented_pairs(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[AugmentedPair]:
    """Stored augmented pairs safe to train on right now: source pair still
    consented, still clears the blocklist, AND still resolves to a train-side
    pair -- re-checked at train time rather than trusted from generation, the
    same belt-and-braces pattern `synthetic.trainable_synthetic_pairs` and
    `synthcorpus.trainable_synthetic_rows` use. A withdrawal or a val/train
    boundary shift (the corpus grew, pairs migrated sides) drops the affected
    variants from training even if they are still sitting in the file.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    train_pairs, _ = _real_train_val_pairs(settings, ids, blocklist)
    train_ids = {p.id for p in train_pairs}
    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    trainable = [
        a
        for a in stored
        if a.source_pair_id in train_ids and not blocklist.matches(a.prompt, a.chosen)
    ]
    return sorted(trainable, key=lambda a: a.id)


def augmented_pair_count(settings: Settings) -> int:
    return AugmentedPairStore(settings.augmented_pairs_path).count()


# --- the leakage check ------------------------------------------------------


class LeakageError(Exception):
    """The leakage check failed: at least one stored augmented pair derives
    from a pair on the *val* side of the split. Raised (not just reported) so
    that anything calling `assert_no_leakage` -- the CLI command and the
    measurement report -- actually fails the run rather than noting the
    problem and moving on."""


@dataclass
class LeakageReport:
    checked: int
    train_side: int
    val_side: int
    leaked: int
    leaked_ids: list[str]
    orphaned: int  # source pair id no longer resolves at all (withdrawn/purged)
    orphaned_ids: list[str]

    @property
    def ok(self) -> bool:
        return self.leaked == 0


def check_leakage(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> LeakageReport:
    """Re-derive the current train/val split of real pairs and check every
    stored augmented pair's `source_pair_id` against it. This recomputes the
    split fresh rather than trusting anything recorded at generation time --
    the split can move (consent withdrawn, corpus grown) between generation
    and any later check, and re-deriving is the only way "no leak" stays true
    as of *now*, not just as of whenever generation ran.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    train_pairs, val_pairs = _real_train_val_pairs(settings, ids, blocklist)
    train_ids = {p.id for p in train_pairs}
    val_ids = {p.id for p in val_pairs}

    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    leaked_ids = sorted({a.id for a in stored if a.source_pair_id in val_ids})
    orphaned_ids = sorted(
        {a.id for a in stored if a.source_pair_id not in train_ids and a.source_pair_id not in val_ids}
    )
    return LeakageReport(
        checked=len(stored),
        train_side=sum(1 for a in stored if a.source_pair_id in train_ids),
        val_side=len(leaked_ids),
        leaked=len(leaked_ids),
        leaked_ids=leaked_ids,
        orphaned=len(orphaned_ids),
        orphaned_ids=orphaned_ids,
    )


def assert_no_leakage(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> LeakageReport:
    """`check_leakage`, but raises `LeakageError` instead of returning a
    report you might forget to inspect. This is what "the run fails loudly"
    means in practice: call this, not `check_leakage`, anywhere a leak must
    stop the pipeline rather than merely appear in a log line."""
    report = check_leakage(settings, ids, blocklist)
    if not report.ok:
        shown = ", ".join(report.leaked_ids[:5]) + (" ..." if len(report.leaked_ids) > 5 else "")
        raise LeakageError(
            f"{report.leaked} augmented pair(s) derive from a val-side source pair: {shown}"
        )
    return report


# --- register / voice-drift check ------------------------------------------

_PUNCT = re.compile(r"[.,!?;:]")
_WORD = re.compile(r"[a-zA-Z']+")


@dataclass
class RegisterReport:
    """How the generated variants' `chosen` text compares to the real
    correction pairs' `chosen` text, on a few surface measures a paraphraser
    drifting toward generic assistant prose would visibly move: length,
    casing, punctuation density, and vocabulary overlap. This is NOT a
    semantic-correctness check (that's what `_parse_variants` and a human
    spot-read are for) -- it is a check that the paraphraser stayed in the
    speaker's *register* while doing so. See the module docstring."""

    real_count: int
    variant_count: int
    real_mean_chars: float
    variant_mean_chars: float
    real_mean_words: float
    variant_mean_words: float
    real_lowercase_rate: float
    variant_lowercase_rate: float
    real_punct_rate: float
    variant_punct_rate: float
    real_vocab_size: int
    variant_vocab_size: int
    vocab_overlap: float  # fraction of variant vocab that also appears in real vocab

    @property
    def drifted(self) -> bool:
        """A blunt, documented threshold rather than a judgement call made
        silently: any one of these firing means the variants no longer read
        like the person being imitated. Thresholds are deliberately loose --
        this flags an obvious problem, it does not certify a subtle one away.
        """
        if self.variant_count == 0:
            return False
        length_ratio = self.variant_mean_chars / max(1.0, self.real_mean_chars)
        return (
            length_ratio > 1.75
            or length_ratio < 0.5
            or (self.real_lowercase_rate - self.variant_lowercase_rate) > 0.15
            or self.vocab_overlap < 0.20
        )


def _text_stats(texts: list[str]) -> tuple[float, float, float, float, set[str]]:
    """mean_chars, mean_words, lowercase_rate, punct_rate, vocab -- shared by
    both sides of `register_comparison` so the two are computed identically."""
    if not texts:
        return 0.0, 0.0, 0.0, 0.0, set()
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    letters = "".join(t for t in texts)
    alpha = [c for c in letters if c.isalpha()]
    lowercase_rate = (sum(1 for c in alpha if c.islower()) / len(alpha)) if alpha else 0.0
    punct_rate = (len(_PUNCT.findall(letters)) / len(letters)) if letters else 0.0
    vocab = {w.lower() for t in texts for w in _WORD.findall(t)}
    return sum(chars) / len(chars), sum(words) / len(words), lowercase_rate, punct_rate, vocab


def register_comparison(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> RegisterReport:
    """Compare every stored augmented pair's `chosen` text against every real
    trainable pair's `chosen` text (the whole real set -- register is a
    property of how the person writes, not specifically of the train split).
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    real_texts = [p.chosen for p in trainable_pairs(settings, ids, blocklist) if p.chosen.strip()]
    variant_texts = [
        a.chosen for a in AugmentedPairStore(settings.augmented_pairs_path).all() if a.chosen.strip()
    ]

    r_chars, r_words, r_lower, r_punct, r_vocab = _text_stats(real_texts)
    v_chars, v_words, v_lower, v_punct, v_vocab = _text_stats(variant_texts)
    overlap = (len(v_vocab & r_vocab) / len(v_vocab)) if v_vocab else 0.0

    return RegisterReport(
        real_count=len(real_texts),
        variant_count=len(variant_texts),
        real_mean_chars=r_chars,
        variant_mean_chars=v_chars,
        real_mean_words=r_words,
        variant_mean_words=v_words,
        real_lowercase_rate=r_lower,
        variant_lowercase_rate=v_lower,
        real_punct_rate=r_punct,
        variant_punct_rate=v_punct,
        real_vocab_size=len(r_vocab),
        variant_vocab_size=len(v_vocab),
        vocab_overlap=overlap,
    )


# --- the auto-fire hook -----------------------------------------------------


class AutoAugmentTrigger:
    """Fires variant generation for exactly the correction pair that was just
    banked -- not a threshold trigger like `AutoPostTrigger`
    (`posttrain.py`), because there is nothing to batch: ro asked for every
    correction to compound the pair set immediately (50 real corrections at
    3 variants each is 200 pairs; another 50 tomorrow makes 400), not for a
    generation run every Nth one.

    Gated on the SAME on/off knob post-train's `--augment-pairs` /
    `Settings.post_augment_pairs` uses -- turning augmentation on turns both
    the training path and this hook on together, and the shipped default
    (off) leaves both off. See the project report for which way the
    with/without measurement went and whether this ships enabled.

    Runs the paraphrase call in a detached subprocess, the same discipline
    `AutoPostTrigger`/`AutoTrainTrigger` use: an LLM round-trip is seconds of
    wall time and must never sit on the bot's reply path. Degrades
    gracefully by construction, not by a try/except bolted on here -- the
    correction row is already written to `interactions.jsonl` by the time
    this fires, and a failed or malformed paraphrase call is caught inside
    `babble augment-pairs` itself (`generate_augmented_pairs`'s per-pair
    try/except around `ParaphraseError`) and only ever costs that one pair's
    variants, never the correction it was trying to augment.
    """

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()

    def on_new_pair(self, pair_id: str) -> None:
        if not self.settings.post_augment_pairs:
            return
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "babble", "augment-pairs",
                    "--pair-id", pair_id, "--quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.log.event("augment.triggered", pid=proc.pid, pair=pair_id)
        except Exception as exc:  # a launch hiccup must never take the bot down
            self.log.event("augment.trigger_failed", error=f"{type(exc).__name__}: {exc}")
