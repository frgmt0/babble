"""The bot's behaviour, driven through the fake Discord layer.

These are the acceptance tests: consent before anything, a decliner leaving no
trace, a correction landing as a complete triple, and a 👍 landing as a weak
positive.
"""

from __future__ import annotations

import json

import pytest

from babble.blocklist import Blocklist
from babble.config import CORRECTION_MARKER
from babble.consent import DECLINED, GRANTED, PENDING, SCOPE_CORPUS, SCOPE_CORRECTIONS, WITHDRAWN
from babble.core import (
    CONSENT_NOTICE,
    CORPUS_NOTICE,
    FOOTER,
    FOOTER_CORPUS_PENDING,
    FOOTER_UNCONSENTED,
    HELP_TEXT,
    Babble,
    IncomingMessage,
    is_correction,
    strip_correction_marker,
)
from babble.corpus import (
    SOURCE_AMBIENT,
    SOURCE_CORRECTION,
    SOURCE_DM,
    SOURCE_MENTION,
    SOURCE_PROMPT,
    SOURCE_REPLY,
)
from babble.store import APPROVAL, CORRECTION
from conftest import FakeDiscord

ALICE = "111111111111111111"
BOB = "222222222222222222"
CAROL = "333333333333333333"

# ro's real Discord channel id, used at least once so the widening flow is
# exercised against the exact id it will actually see in production.
RO_CHANNEL = "1400209071017169116"


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


# --- silent drops are logged, not silent ---------------------------------


def test_a_message_not_addressed_to_it_is_logged_as_a_drop(fake, read_log):
    fake.say(ALICE, "just chatting to my friends")

    (event,) = read_log("bot.dropped")
    assert event["reason"] == "not_addressed"
    assert event["user"].startswith("u_")
    assert event["channel"].startswith("c_")


def test_a_dropped_message_never_leaks_its_content_into_the_log(fake, read_log, settings):
    fake.say(ALICE, "a secret nobody has consented to store")

    log_text = (settings.log_dir / "babble.jsonl").read_text()
    assert "a secret nobody has consented to store" not in log_text
    assert ALICE not in log_text


def test_a_message_from_another_bot_is_dropped_and_logged(brain, read_log):
    replies = brain.handle_message(
        IncomingMessage(message_id="1", author_id="other-bot", content="hi", author_is_bot=True)
    )

    assert replies == []
    (event,) = read_log("bot.dropped")
    assert event["reason"] == "author_is_bot"


def test_a_ping_records_which_guild_it_came_from(brain, read_log):
    brain.handle_message(
        IncomingMessage(
            message_id="1",
            author_id=ALICE,
            content="hi",
            channel_id="chan-1",
            guild_id="guild-42",
            mentions_bot=True,
        )
    )

    (event,) = read_log("bot.ping")
    assert event["guild"].startswith("g_")
    assert "guild-42" not in json.dumps(event)


def test_a_dm_ping_has_no_guild_field_at_all(brain, read_log):
    brain.handle_message(
        IncomingMessage(
            message_id="1", author_id=ALICE, content="hi", channel_id="chan-1", mentions_bot=True
        )
    )

    (event,) = read_log("bot.ping")
    assert "guild" not in event


def test_declining_user_still_gets_replies_but_nothing_is_ever_stored(fake, brain):
    fake.ping(BOB)
    fake.decline(BOB)

    answer = fake.ping(BOB, "hello there")[0]
    assert answer.kind == "generation"
    assert FOOTER_UNCONSENTED in answer.content

    # Even trying to correct it stores nothing.
    fake.correct(BOB, "you should have said hi", reply_to=answer.id)
    fake.react(BOB, answer.id)

    assert brain.consent.decision(BOB) == DECLINED
    assert brain.store.count() == 0
    assert len(brain.exchanges) == 0  # not even cached pending


def test_user_who_ignores_the_notice_gets_replies_but_no_capture(fake, brain):
    fake.ping(CAROL)  # notice; never answered
    answer = fake.ping(CAROL, "hello?")[0]

    assert answer.kind == "generation"
    assert FOOTER_UNCONSENTED in answer.content
    fake.correct(CAROL, "wrong, say hi", reply_to=answer.id)
    assert brain.store.count() == 0


# --- corrections --------------------------------------------------------


def test_correction_lands_as_a_complete_triple(fake, brain, generator):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    assert FOOTER in answer.content

    ack = fake.correct(ALICE, "you should say: hey!", reply_to=answer.id)[0]

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

    fake.correct(ALICE, gif, reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == gif


def test_an_uploaded_gif_is_captured_via_its_attachment_url(fake, brain):
    url = "https://cdn.discordapp.com/attachments/1/2/reaction.gif"
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "react to this")[0]

    fake.correct(ALICE, "like this", reply_to=answer.id, attachments=(url,))

    (row,) = brain.store.all()
    assert url in row.chosen


def test_emoji_only_correction_is_kept(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "how do you feel")[0]

    fake.correct(ALICE, "🫠", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == "🫠"


def test_correction_needs_consent_from_everyone_in_the_thread(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(BOB, "actually say hi")  # Bob's first contact: gets the notice
    fake.decline(BOB)
    ack = fake.correct(BOB, "actually say hi", reply_to=answer.id)[0]

    assert brain.store.count() == 0
    assert "not stored" in ack.content


def test_the_same_correction_twice_is_one_row(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)
    fake.correct(ALICE, "say hey", reply_to=answer.id)

    assert brain.store.count() == 1


class _FakePostTrigger:
    def __init__(self) -> None:
        self.calls = 0

    def maybe_run(self) -> None:
        self.calls += 1


def test_a_fresh_correction_pokes_the_post_trigger(settings, generator, log):
    """A correction landing is exactly what should offer the post-train
    trigger a chance to fire -- mirrors how a fresh corpus row pokes
    `train_trigger`."""
    post_trigger = _FakePostTrigger()
    brain = Babble(settings, generator=generator, log=log, post_trigger=post_trigger)
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)

    assert post_trigger.calls == 1


def test_a_duplicate_correction_does_not_poke_the_post_trigger(settings, generator, log):
    post_trigger = _FakePostTrigger()
    brain = Babble(settings, generator=generator, log=log, post_trigger=post_trigger)
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)
    fake.correct(ALICE, "say hey", reply_to=answer.id)  # same pair -> not fresh

    assert post_trigger.calls == 1


class _FakeAugmentTrigger:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_new_pair(self, pair_id: str) -> None:
        self.calls.append(pair_id)


def test_a_fresh_correction_pokes_the_augment_trigger_with_its_own_id(settings, generator, log):
    """Unlike the post-train trigger, augmentation fires per-correction, not
    on a threshold -- and it must be told exactly which pair to paraphrase."""
    augment_trigger = _FakeAugmentTrigger()
    brain = Babble(settings, generator=generator, log=log, augment_trigger=augment_trigger)
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)

    assert len(augment_trigger.calls) == 1
    assert augment_trigger.calls[0] == brain.store.all()[0].id


def test_a_duplicate_correction_does_not_poke_the_augment_trigger(settings, generator, log):
    augment_trigger = _FakeAugmentTrigger()
    brain = Babble(settings, generator=generator, log=log, augment_trigger=augment_trigger)
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)
    fake.correct(ALICE, "say hey", reply_to=answer.id)  # same pair -> not fresh

    assert len(augment_trigger.calls) == 1


def test_no_augment_trigger_is_a_silent_no_op(settings, generator, log):
    """Same contract as `post_trigger=None`: a brain built without one (every
    existing test, and any deployment that never wires it in) works exactly
    as before."""
    brain = Babble(settings, generator=generator, log=log)
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.correct(ALICE, "say hey", reply_to=answer.id)  # must not raise

    assert brain.store.count() == 1


# --- the correction marker -----------------------------------------------
#
# Before this, every reply to one of the bot's messages was stored as the answer
# it should have given -- so "lol", "wrong" and "what" all went into the corpus
# as lessons. A correction now has to say it is one.


def test_a_marked_reply_is_stored_with_the_marker_stripped(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, f"{CORRECTION_MARKER} hey, what's up", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.signal == CORRECTION
    assert row.chosen == "hey, what's up", "the marker must never reach the corpus"
    assert CORRECTION_MARKER not in row.chosen


def test_the_marker_is_stripped_with_or_without_a_space_after_it(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, f"{CORRECTION_MARKER}hey", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == "hey"


def test_only_one_space_after_the_marker_is_eaten(fake, brain):
    """Anything past the first space is the person's own formatting."""
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, f"{CORRECTION_MARKER}  spaced out", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.chosen == "spaced out"


def test_an_unmarked_reply_is_answered_not_stored(fake, brain, generator, read_log):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    replies = fake.ping(ALICE, "lol what", reply_to=answer.id)

    assert brain.store.count() == 0, "an unmarked reply is not a lesson"
    # It is still a message addressed to the bot, so it gets answered.
    assert replies and replies[0].kind == "generation"
    assert generator.text in replies[0].content
    assert read_log("capture.unmarked"), "declining to learn from a reply must be recorded"
    assert not read_log("bot.dropped"), "the message was answered, so it was not dropped"


def test_an_unmarked_reply_is_never_filed_as_a_correction_pair_even_from_a_consenting_user(
    fake, brain
):
    """It still lands in the text corpus as an ordinary reply -- see the corpus
    capture tests below -- it just isn't treated as teaching a lesson."""
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, "you should have said hi", reply_to=answer.id)
    fake.ping(ALICE, ">nearly the marker", reply_to=answer.id)

    assert brain.store.count() == 0


def test_the_marker_alone_is_rejected_with_the_format(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    ack = fake.ping(ALICE, f"{CORRECTION_MARKER}   ", reply_to=answer.id)[0]

    assert brain.store.count() == 0
    assert ack.kind == "ack"
    assert CORRECTION_MARKER in ack.content, "tell them the format they got wrong"


def test_a_thumbs_up_still_works_without_any_marker(fake, brain, generator):
    """The marker is a reply-path rule. Reactions are untouched."""
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.react(ALICE, answer.id)

    (row,) = brain.store.all()
    assert row.signal == APPROVAL
    assert row.rejected is None
    assert row.chosen == generator.text


def test_the_footer_and_help_both_teach_the_marker():
    assert CORRECTION_MARKER in FOOTER
    assert CORRECTION_MARKER in HELP_TEXT
    assert CORRECTION_MARKER in CONSENT_NOTICE


def test_marker_helpers_agree_on_what_counts():
    assert is_correction(f"{CORRECTION_MARKER} hey")
    assert is_correction(f"   {CORRECTION_MARKER}hey")  # leading whitespace is fine
    assert not is_correction("hey")
    assert not is_correction(f"hey {CORRECTION_MARKER}")  # must *begin* with it
    assert strip_correction_marker(f"{CORRECTION_MARKER} hey") == "hey"
    assert strip_correction_marker(f"{CORRECTION_MARKER}") == ""


# --- the corpus: what actually trains the model --------------------------


def test_a_consented_mention_lands_in_the_corpus_under_the_senders_pseudonym(fake, brain):
    fake.onboard(ALICE)

    fake.ping(ALICE, "what a nice day for it")

    (row,) = brain.corpus.all()
    assert row.text == "what a nice day for it"
    assert row.author.startswith("u_")
    assert row.source == SOURCE_MENTION


def test_a_reply_to_the_bot_is_filed_with_source_reply(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]

    fake.ping(ALICE, "lol what", reply_to=answer.id)

    (row,) = [r for r in brain.corpus.all() if r.text == "lol what"]
    assert row.source == SOURCE_REPLY


def test_a_dm_is_filed_with_source_dm_and_no_guild(fake, brain):
    fake.onboard(ALICE)

    fake.dm(ALICE, "just the two of us")

    (row,) = [r for r in brain.corpus.all() if r.text == "just the two of us"]
    assert row.source == SOURCE_DM
    assert row.guild is None


def test_the_bots_own_generated_text_never_lands_in_the_corpus(fake, brain, generator):
    fake.onboard(ALICE)

    fake.ping(ALICE, "hello")

    texts = [row.text for row in brain.corpus.all()]
    assert generator.text not in texts


def test_a_legacy_corrections_only_consenter_keeps_getting_answered_but_not_collected(
    fake, brain
):
    """Someone who granted only the old `corrections` scope is a full participant
    for corrections, but their plain messages wait for the new notice -- shown
    exactly once, not on every ping."""
    brain.consent.grant(ALICE, SCOPE_CORRECTIONS)

    first = fake.ping(ALICE, "hello")
    assert [r.kind for r in first] == ["generation", "consent"]
    assert FOOTER_CORPUS_PENDING in first[0].content
    assert first[1].content == CORPUS_NOTICE
    assert brain.corpus.count() == 0

    second = fake.ping(ALICE, "hello again")
    assert [r.kind for r in second] == ["generation"], "the re-ask is shown only once"
    assert brain.corpus.count() == 0

    fake.accept(ALICE)
    fake.ping(ALICE, "now I'm all the way in")

    assert brain.corpus.count() == 1


def test_babble_all_collects_everything_said_in_that_channel_as_ambient_with_no_reply(
    fake, brain
):
    """Ro's real channel id, so the widening flow is exercised end to end
    against the exact id it sees in production."""
    fake.onboard(ALICE)

    fake.collect_all(ALICE, channel=RO_CHANNEL)
    replies = fake.say(ALICE, "just chatting, not pinging anybody", channel=RO_CHANNEL)

    assert replies == []
    (row,) = brain.corpus.all()
    assert row.text == "just chatting, not pinging anybody"
    assert row.source == SOURCE_AMBIENT


def test_widening_does_not_follow_the_same_person_into_a_different_channel(fake, brain):
    fake.onboard(ALICE)
    fake.collect_all(ALICE, channel="chan-1")

    fake.say(ALICE, "said somewhere else entirely", channel="chan-2")

    assert brain.corpus.count() == 0


def test_widening_does_not_affect_anybody_else_in_the_same_channel(fake, brain):
    fake.onboard(ALICE)
    fake.onboard(BOB)
    fake.collect_all(ALICE, channel="chan-1")

    fake.say(BOB, "bob never ran babble all", channel="chan-1")

    assert brain.corpus.count() == 0


def test_babble_pings_turns_ambient_capture_off_from_the_very_next_message(fake, brain):
    fake.onboard(ALICE)
    fake.collect_all(ALICE)
    fake.say(ALICE, "captured while it's on")

    fake.only_pings(ALICE)
    fake.say(ALICE, "not captured anymore")

    texts = [row.text for row in brain.corpus.all()]
    assert texts == ["captured while it's on"]


def test_babble_all_from_someone_who_has_not_opted_in_shows_the_consent_notice_and_widens_nothing(
    fake, brain
):
    reply = fake.collect_all(BOB)[0]

    assert CONSENT_NOTICE in reply.content
    assert brain.consent.wide_channels(BOB) == []

    fake.say(BOB, "still never opted in")

    assert brain.corpus.count() == 0


def test_forget_purges_the_corpus_too_and_clears_widened_channels(fake, brain):
    fake.onboard(ALICE)
    fake.collect_all(ALICE)
    fake.say(ALICE, "ambient text of mine")
    answer = fake.ping(ALICE, "hello")[0]
    fake.correct(ALICE, "say hey", reply_to=answer.id)
    assert brain.corpus.count() > 0
    assert brain.consent.wide_channels(ALICE) == ["chan-1"]

    fake.say(ALICE, "!babble forget")

    assert brain.corpus.count() == 0
    assert brain.consent.wide_channels(ALICE) == []


def test_blocklisted_text_never_reaches_the_corpus_from_an_addressed_message(
    settings, generator, log
):
    brain = Babble(
        settings, generator=generator, log=log, blocklist=Blocklist(frozenset({"badword"}))
    )
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)

    gw.ping(ALICE, "this has a badword right in it")

    assert brain.corpus.count() == 0


def test_blocklisted_text_never_reaches_the_corpus_from_an_ambient_message(
    settings, generator, log
):
    brain = Babble(
        settings, generator=generator, log=log, blocklist=Blocklist(frozenset({"badword"}))
    )
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)
    gw.collect_all(ALICE)

    gw.say(ALICE, "an ambient badword slipping through")

    assert brain.corpus.count() == 0


def test_a_correction_files_corpus_rows_for_both_halves_under_their_own_authors(fake, brain):
    """The corrector's text and the original prompt each need their own
    author's corpus grant -- one having it doesn't cover the other."""
    brain.consent.grant(ALICE, SCOPE_CORRECTIONS)  # legacy: hasn't agreed to the corpus yet
    answer = fake.ping(ALICE, "an original prompt")[0]  # too early to be collected
    brain.consent.grant(ALICE, SCOPE_CORPUS)  # accepts the corpus before the correction lands
    fake.onboard(BOB)

    fake.correct(BOB, "a much better answer", reply_to=answer.id)

    by_text = {row.text: row for row in brain.corpus.all()}
    assert by_text["a much better answer"].source == SOURCE_CORRECTION
    assert by_text["an original prompt"].source == SOURCE_PROMPT
    assert by_text["a much better answer"].author != by_text["an original prompt"].author


def test_a_correction_from_someone_without_a_corpus_grant_only_files_the_prompt_half(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "an original prompt")[0]
    brain.consent.grant(BOB, SCOPE_CORRECTIONS)  # legacy: corrections yes, corpus unknown

    fake.correct(BOB, "a much better answer", reply_to=answer.id)

    (row,) = brain.store.all()
    assert row.signal == CORRECTION  # the correction pair is filed regardless

    texts = {r.text for r in brain.corpus.all()}
    assert "an original prompt" in texts
    assert "a much better answer" not in texts, "bob never granted the corpus scope"


def test_a_message_not_addressed_to_the_bot_is_still_dropped_when_nobody_has_widened(
    fake, brain, read_log
):
    fake.onboard(ALICE)  # consented, but never ran `!babble all`

    replies = fake.say(ALICE, "just chatting, not pinging the bot")

    assert replies == []
    assert brain.corpus.count() == 0
    (event,) = read_log("bot.dropped")
    assert event["reason"] == "not_addressed"


def test_babble_commands_are_never_stored_in_the_corpus(fake, brain):
    fake.onboard(ALICE)  # itself sent as `!babble accept`

    fake.say(ALICE, "!babble status")
    fake.say(ALICE, "!babble consent")
    fake.collect_all(ALICE)

    assert brain.corpus.count() == 0


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
    fake.correct(ALICE, "say hey", reply_to=answer.id)
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
    fake.correct(ALICE, "say hey", reply_to=a_answer.id)
    b_answer = fake.ping(BOB, "hi")[0]
    fake.correct(BOB, "say yo", reply_to=b_answer.id)
    assert brain.store.count() == 2

    fake.say(ALICE, "!babble forget")

    (survivor,) = brain.store.all()
    assert survivor.chosen == "say yo"


def test_a_withdrawn_user_can_opt_back_in(fake, brain):
    fake.onboard(ALICE)
    fake.say(ALICE, "!babble forget")

    fake.accept(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.correct(ALICE, "say hey", reply_to=answer.id)

    assert brain.store.count() == 1


# --- identifiers --------------------------------------------------------


def test_discord_mentions_are_scrubbed_out_of_stored_text(fake, brain):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "<@bot-9999> what about <@444444444444444444>?")[0]

    fake.correct(ALICE, "tell <@555555555555555555> hello", reply_to=answer.id)

    (row,) = brain.store.all()
    assert "444444444444444444" not in row.prompt
    assert "555555555555555555" not in row.chosen
    assert "@user" in row.chosen


def test_no_raw_discord_id_reaches_the_dataset_or_the_logs(fake, brain, settings):
    fake.onboard(ALICE)
    answer = fake.ping(ALICE, "hello")[0]
    fake.correct(ALICE, "say hey", reply_to=answer.id)
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
    assert CORRECTION_MARKER in content, "the footer must teach the correction format"


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
    gw.correct(ALICE, "say hey", reply_to=answer.id)
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
    reborn.correct(ALICE, "you should say hey", reply_to=answer.id)

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
