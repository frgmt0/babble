"""A dozen made-up corrections, so the trainer has something to chew on offline.

This is **not** real data and **not** a seed for the real corpus. It exists so
that `babble train` can be run and watched before the bot has a token, and so the
tests have realistic rows. It is written by an explicit command, into the same
stores the bot writes to, attributed to fake users who "consented" locally.

Both stores get filled: the correction pairs go in as pairs, and then the same
backfill the real migration uses flattens them into corpus rows, because the
corpus is what the trainer reads.

If you are running this for real: delete `data/` before you go live, or these
lines will end up in your export looking like things a person said.
"""

from __future__ import annotations

from .config import Settings
from .consent import ConsentStore
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .store import APPROVAL, CORRECTION, Interaction, InteractionStore, make_row_id
from .util import utcnow_iso

# (what someone said, what the bot got wrong, what it should have said)
FAKE_ROWS = [
    ("hello", "qX7`~ tt", "hey!"),
    ("hey", "]]9 zzz", "hi :)"),
    ("who are you", "\x01\x02 gg", "babble. i'm learning to talk."),
    ("what are you", "vvv 88", "a very small model."),
    ("good morning", "mmm ?!", "morning!"),
    ("how are you", "kkk 12", "still mostly noise, thanks."),
    ("say something", "~~~", "something."),
    ("post a gif", "http mmm", "https://tenor.com/view/cat-typing-1234"),
    ("thanks", "ppp", "np"),
    ("goodnight", "zz9", "night!"),
    ("what is your name", "bbb??", "babble"),
    ("are you real", "yyy 00", "no, i'm mostly random numbers."),
]

FAKE_USER = "fake-tester-000"
FAKE_HELPER = "fake-tester-001"


def seed_fake_data(
    settings: Settings,
    log: EventLog | None = None,
    user_id: str = FAKE_USER,
    helper_id: str = FAKE_HELPER,
) -> int:
    """Write the fake rows. Returns how many were new."""
    settings.ensure_dirs()
    log = log or NullLog()
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    store = InteractionStore(settings.interactions_path)

    # These rows only count as trainable if the fake users have consented, which
    # is exactly the rule real rows follow.
    consent.grant(user_id)
    consent.grant(helper_id)
    asker, helper = ids.user(user_id), ids.user(helper_id)

    added = 0
    for prompt, rejected, chosen in FAKE_ROWS:
        row = Interaction(
            id=make_row_id(CORRECTION, prompt, chosen, asker, helper),
            signal=CORRECTION,
            prompt=prompt,
            rejected=rejected,
            chosen=chosen,
            prompt_author=asker,
            signal_author=helper,
            weight=settings.correction_weight,
            created_at=utcnow_iso(),
        )
        added += int(store.append(row))

    # One approval, so the weighting path has something to exercise too.
    approval = Interaction(
        id=make_row_id(APPROVAL, "hello", "hey!", asker, helper),
        signal=APPROVAL,
        prompt="hello",
        rejected=None,
        chosen="hey!",
        prompt_author=asker,
        signal_author=helper,
        weight=settings.approval_weight,
        created_at=utcnow_iso(),
    )
    added += int(store.append(approval))

    # The trainer reads the corpus, not the pairs, so seeding pairs alone would
    # leave `babble train` with nothing to do. Same one-shot migration the real
    # data goes through, so what gets trained here is shaped like the real thing.
    from .backfill import backfill_corpus  # local: keeps this module import-cheap
    from .corpus import CorpusStore

    backfilled = backfill_corpus(settings, log=log, ids=ids)

    log.event(
        "fakedata.seed",
        added=added,
        total=store.count(),
        corpus_added=backfilled.added,
        # How big the corpus actually is, not how much this backfill walked --
        # a real deployment has rows in there that no correction ever produced.
        corpus_total=CorpusStore(settings.corpus_path).count(),
    )
    return added
