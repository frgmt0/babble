"""The bot's behaviour, driven through the fake Discord layer.

These are the acceptance tests: consent before anything, a decliner leaving no
trace, a correction landing as a complete triple, and a 👍 landing as a weak
positive.
"""

from __future__ import annotations

import pytest

from babble.consent import DECLINED, GRANTED, PENDING, WITHDRAWN
from babble.core import FOOTER, FOOTER_UNCONSENTED
from babble.store import APPROVAL, CORRECTION

ALICE = "111111111111111111"
BOB = "222222222222222222"
CAROL = "333333333333333333"


# --- consent ------------------------------------------------------------


def test_first_time_pinger_gets_the_consent_notice(fake, brain):
    posted = fake.ping(ALICE, "hey bot")

    assert len(posted) == 1
    assert posted[0].kind == "consent"
    assert "!babble accept" in posted[0].content
    assert "HuggingFace" in posted[0].content
    # Nothing is generated and nothing is kept on first contact.
    assert brain.store.count() == 0
    assert brain.consent.decision(ALICE) == PENDING


def test_notice_is_shown_once_then_it_just_answers(fake, brain, generator):
    fake.ping(ALICE)
    fake.accept(ALICE)

    posted = fake.ping(ALICE, "hello again")

    assert posted[0].kind == "generation"
    assert generator.text in posted[0].content
    assert brain.consent.decision(ALICE) == GRANTED


def test_it_ignores_messages_that_are_not_addressed_to_it(fake, brain):
    assert fake.say(ALICE, "just chatting to my friends") == []
    assert brain.consent.decision(ALICE) == "unknown"


def test_declining_user_still_gets_replies_but_nothing_is_ever_stored(fake, brain):
    fake.ping(BOB)
    fake.decline(BOB)

    answer = fake.ping(BOB, "hello there")[0]
    assert answer.kind == "generation"
    assert FOOTER_UNCONSENTED in answer.content

    # Even trying to correct it stores nothing.
    fake.ping(BOB, "you should have said hi", reply_to=answer.id)
    fake.react(BOB, answer.id)

    assert brain.consent.decision(BOB) == DECLINED
    assert brain.store.count() == 0
    assert len(brain.exchanges) == 0  # not even cached pending


def test_user_who_ignores_the_notice_gets_replies_but_no_capture(fake, brain):
    fake.ping(CAROL)  # notice; never answered
    answer = fake.ping(CAROL, "hello?")[0]

    assert answer.kind == "generation"
    assert FOOTER_UNCONSENTED in answer.content
    fake.ping(CAROL, "wrong, say hi", reply_to=answer.id)
    assert brain.store.count() == 0


# --- corrections --------------------------------------------------------


def test_correction_lands_as_a_complete_triple(fake, brain, generator):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    assert FOOTER in answer.content

    ack = fake.ping(ALICE, "you should say: hey!", reply_to=answer.id)[0]

    (row,) = brain.store.all()
    assert row.signal == CORRECTION
    assert row.prompt == "hello"
    assert row.rejected == generator.text
    assert row.chosen == "you should say: hey!"
    assert row.weight == brain.settings.correction_weight
    assert row.prompt_author.startswith("u_")
    assert row.signal_author.startswith("u_")
    assert ack.kind == "ack"
    assert "correction #1" in ack.content


def test_a_gif_url_correction_is_stored_exactly_as_typed(fake, brain):
    gif = "https://tenor.com/view/cat-typing-furiously-98765"
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "post a gif")[0]

    fake.ping(ALICE, gif, reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == gif


def test_an_uploaded_gif_is_captured_via_its_attachment_url(fake, brain):
    url = "https://cdn.discordapp.com/attachments/1/2/reaction.gif"
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "react to this")[0]

    fake.ping(ALICE, "like this", reply_to=answer.id, attachments=(url,))

    (row,) = brain.store.all()
    assert url in row.chosen


def test_emoji_only_correction_is_kept(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "how do you feel")[0]

    fake.ping(ALICE, "🫠", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == "🫠"


def test_correction_needs_consent_from_everyone_in_the_thread(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(BOB, "actually say hi")  # Bob's first contact: gets the notice
    fake.decline(BOB)
    ack = fake.ping(BOB, "actually say hi", reply_to=answer.id)[0]

    assert brain.store.count() == 0
    assert "not stored" in ack.content


def test_the_same_correction_twice_is_one_row(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, "say hey", reply_to=answer.id)
    fake.ping(ALICE, "say hey", reply_to=answer.id)

    assert brain.store.count() == 1


def test_empty_reply_is_not_stored_as_a_correction(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    ack = fake.ping(ALICE, "   ", reply_to=answer.id)[0]

    assert brain.store.count() == 0
    assert "nothing to learn" in ack.content


# --- reactions ----------------------------------------------------------


def test_thumbs_up_lands_as_a_positive_signal(fake, brain, generator):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    assert fake.react(ALICE, answer.id, "👍") == []  # silent, no channel spam

    (row,) = brain.store.all()
    assert row.signal == APPROVAL
    assert row.prompt == "hello"
    assert row.rejected is None
    assert row.chosen == generator.text


def test_a_thumbs_up_counts_for_less_than_a_correction(fake, brain, settings):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.react(ALICE, answer.id)

    (row,) = brain.store.all()
    assert row.weight == settings.approval_weight
    assert row.weight < settings.correction_weight


def test_thumbs_up_with_a_variation_selector_still_counts(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.react(ALICE, answer.id, "👍️")

    assert brain.store.count() == 1


@pytest.mark.parametrize("emoji", ["👎", "🔥", "❤️"])
def test_other_reactions_are_not_signals(fake, brain, emoji):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.react(ALICE, answer.id, emoji)

    assert brain.store.count() == 0


def test_thumbs_up_from_someone_who_never_consented_is_dropped(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.react(BOB, answer.id)  # Bob has never been asked

    assert brain.store.count() == 0


def test_thumbs_up_on_a_message_we_dont_know_is_ignored(fake, brain):
    fake.onboard(ALICE)

    fake.react(ALICE, "some-old-message-id")

    assert brain.store.count() == 0


# --- withdrawal ---------------------------------------------------------


def test_forget_withdraws_consent_and_purges_stored_rows(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.ping(ALICE, "say hey", reply_to=answer.id)
    assert brain.store.count() == 1

    reply = fake.say(ALICE, "!babble forget")[0]

    assert brain.store.count() == 0
    assert brain.consent.decision(ALICE) == WITHDRAWN
    assert "1" in reply.content
    assert len(brain.exchanges) == 0  # pending exchanges go too


def test_forget_only_purges_that_persons_rows(fake, brain):
    fake.onboard(ALICE)
    fake.onboard(BOB)
    a_answer = fake.ping(ALICE, "hello")[0]
    fake.ping(ALICE, "say hey", reply_to=a_answer.id)
    b_answer = fake.ping(BOB, "hi")[0]
    fake.ping(BOB, "say yo", reply_to=b_answer.id)
    assert brain.store.count() == 2

    fake.say(ALICE, "!babble forget")

    (survivor,) = brain.store.all()
    assert survivor.chosen == "say yo"


def test_a_withdrawn_user_can_opt_back_in(fake, brain):
    fake.onboard(ALICE)
    fake.say(ALICE, "!babble forget")

    fake.accept(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.ping(ALICE, "say hey", reply_to=answer.id)

    assert brain.store.count() == 1


# --- identifiers --------------------------------------------------------


def test_discord_mentions_are_scrubbed_out_of_stored_text(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "<@bot-9999> what about <@444444444444444444>?")[0]

    fake.ping(ALICE, "tell <@555555555555555555> hello", reply_to=answer.id)

    (row,) = brain.store.all()
    assert "444444444444444444" not in row.prompt
    assert "555555555555555555" not in row.chosen
    assert "@user" in row.chosen


def test_no_raw_discord_id_reaches_the_dataset_or_the_logs(fake, brain, settings):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.ping(ALICE, "say hey", reply_to=answer.id)
    fake.react(ALICE, answer.id)
    brain.log.flush()

    published = settings.interactions_path.read_text(encoding="utf-8")
    events = (settings.log_dir / "babble.jsonl").read_text(encoding="utf-8")
    prose = (settings.log_dir / "babble.log").read_text(encoding="utf-8")

    for blob in (published, events, prose):
        assert ALICE not in blob

    # ...but the local consent record has to know who it is talking about.
    assert ALICE in settings.consent_path.read_text(encoding="utf-8")


# --- commands and copy --------------------------------------------------


def test_every_generation_carries_the_feedback_footer(fake):
    fake.onboard(ALICE)

    content = fake.ping(ALICE, "hello")[0].content

    assert "-#" in content
    assert "👍" in content
    assert "correct me with a reply" in content


def test_help_and_status_answer_without_touching_the_model(fake, generator):
    assert "!babble consent" in fake.say(ALICE, "!babble")[0].content
    assert "step" in fake.say(ALICE, "!babble status")[0].content
    assert "stored" in fake.say(ALICE, "!babble consent")[0].content
    assert generator.prompts == []


def test_a_message_merely_starting_with_babble_is_not_a_command(fake, brain):
    fake.onboard(ALICE)

    fake.ping(ALICE, "!babbles are what you do underwater")

    assert brain.store.count() == 0
    assert fake.last.kind == "generation"


def test_the_bot_never_answers_another_bot(brain):
    from babble.core import IncomingMessage

    replies = brain.handle_message(
        IncomingMessage(
            message_id="1", author_id="other-bot", content="hi", author_is_bot=True, mentions_bot=True
        )
    )

    assert replies == []


# --- construction without a logger ---------------------------------------


def test_the_brain_works_with_no_log_attached(settings, generator):
    """NullLog is the default; every path must survive it."""
    from babble.core import Babble
    from conftest import FakeDiscord

    brain = Babble(settings, generator=generator)
    gw = FakeDiscord(brain)

    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    gw.ping(ALICE, "say hey", reply_to=answer.id)
    gw.react(ALICE, answer.id)

    # Correcting an answer and then also 👍-ing it is two distinct facts.
    assert sorted(r.signal for r in brain.store.all()) == [APPROVAL, CORRECTION]


def test_every_message_the_bot_sends_fits_in_a_discord_message(fake):
    from babble.core import DISCORD_LIMIT

    fake.ping(ALICE)  # consent notice, the longest thing it ever sends
    fake.accept(ALICE)
    fake.say(ALICE, "!babble consent")
    fake.say(ALICE, "!babble help")
    fake.say(ALICE, "!babble status")
    fake.ping(ALICE, "hello")

    for message in fake.sent:
        assert len(message.content) <= DISCORD_LIMIT, message.kind


def test_a_rambling_generation_is_truncated_to_fit_with_its_footer(settings, log):
    from babble.core import FOOTER, DISCORD_LIMIT, Babble
    from conftest import FakeDiscord

    brain = Babble(settings, generator=lambda p: "y" * 5000, log=log)
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)

    content = gw.ping(ALICE, "hello")[0].content

    assert len(content) <= DISCORD_LIMIT
    assert content.endswith(FOOTER)


def test_output_that_is_pure_control_bytes_still_produces_a_message(settings, log):
    from babble.core import Babble
    from conftest import FakeDiscord

    brain = Babble(settings, generator=lambda p: "\x00\x01\x02", log=log)
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)

    content = gw.ping(ALICE, "hello")[0].content

    assert content.strip()  # Discord rejects empty messages
    assert "noise" in content


# --- surviving a restart -------------------------------------------------


def test_a_correction_still_lands_after_the_bot_reboots(settings, generator, log):
    """The exchange log is on disk precisely so this works."""
    from babble.core import Babble
    from conftest import FakeDiscord

    before = Babble(settings, generator=generator, log=log)
    gw = FakeDiscord(before)
    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]

    # Whole new process, same directory: fresh stores, nothing in memory.
    after = Babble(settings, generator=generator, log=log)
    reborn = FakeDiscord(after)
    reborn.ping(ALICE, "you should say hey", reply_to=answer.id)

    (row,) = after.store.all()
    assert row.prompt == "hello"
    assert row.rejected == generator.text
    assert row.chosen == "you should say hey"


def test_consent_survives_a_restart_too(settings, generator, log):
    from babble.core import Babble
    from conftest import FakeDiscord

    FakeDiscord(Babble(settings, generator=generator, log=log)).onboard(ALICE)

    after = Babble(settings, generator=generator, log=log)
    posted = FakeDiscord(after).ping(ALICE, "hello")

    assert posted[0].kind == "generation"  # not asked to consent all over again


def test_the_core_logic_does_not_depend_on_discord():
    """bot.py is the only place discord.py may appear."""
    import ast
    import pathlib

    source = pathlib.Path(__import__("babble.core", fromlist=["core"]).__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert "discord" not in imported
    assert "torch" not in imported  # and it stays cheap to import
