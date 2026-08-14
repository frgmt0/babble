# Collection feed: show data coming in, not training
> run: run-20260814-collection-feed-show-data-coming-in-not · branch: beckett/run-collection-feed-show-data-coming-in-not · created: 2026-08-14T22:11:41.238Z

## Goal
ro (user 1151230208783945818, the owner) asked:

"Right so the training loop needs to stop because we are just in pretraining not post training RL
yet. #1537688090217418773 should now become the channel that shows when data is added not being
trained"

Channel 1537688090217418773 is the Discord channel the training feed currently posts to. Babble
just pivoted from correction pairs to collecting an unlabelled pretraining corpus; the corpus is
tiny (54 rows, ~1k characters) and training it continuously is pointless until there is real data.

I have ALREADY stopped the live trainer and restarted the live bot on current main. Do not try to
start, stop or touch anything running — the live deployment is `~/babble-live` and is not yours.
Your job is the code in this repo (`kowo-co/babble`, main at `a088bc1`).

## What to build

### 1. The channel becomes a collection feed

`babble/discord_feed.py` currently has a `TrainingFeed` that posts cycle-start and checkpoint
messages. Training is not running, so that channel is now silent. It needs to report **collection**
instead — the moments when the corpus changes:

- **A corpus row was added.** Show the text that was collected (blocklist-filtered and neutered
  exactly as sample text already is), which surface it came from (a ping at booper, a DM, or a
  widened `!babble all` channel grant), the contributor's PSEUDONYM only, and the running totals:
  corpus rows, characters, distinct contributors.
- **Consent changed**, because that is what changes what may be collected at all: someone granted
  the default consent, someone opted a channel in with the widened grant, someone revoked it,
  someone withdrew entirely (and how many rows that purge removed).
- **Milestones**, so growth is legible without reading every line: a short message every N rows and
  every N characters. Pick sensible N for a corpus that is currently 54 rows — something that fires
  meaningfully now and does not become spam at 10,000 rows. Scale the interval with size rather
  than using one fixed number.

**Do not spam the channel.** One message per row is acceptable at the current trickle, but someone
who has opted a channel in with `!babble all` can produce a burst. Coalesce rows arriving inside a
short window (a few seconds) into one message that lists them, rather than posting each separately.

Keep the existing `TrainingFeed` code working and intact — it is correct, it just has nothing to
report while no trainer is running. When a trainer IS run manually, its messages should still post.
This is an ADDITION, not a replacement.

### 2. Training is opt-in, never ambient

Make sure nothing starts a training loop on its own: no autostart on bot boot, no implicit loop.
`babble train` stays a deliberate command someone runs. If anything in the repo currently implies
or documents continuous training as the default mode, fix it — including README and spec.

### 3. The HF publish cadence — this breaks silently otherwise

The auto-publish to HuggingFace currently fires every 20 CHECKPOINTS. With no trainer running there
are no checkpoints, so the public dataset would quietly never update again — which is the opposite
of what a collection phase wants.

Re-key the publish cadence to DATA rather than to training: publish when the corpus has grown by a
meaningful amount since the last publish (rows or characters, your call, state it in the code), with
the same consent and blocklist re-checks at export that already exist. It must still be impossible
to publish a row whose author has not consented or whose text matches the blocklist. Announce the
publish in the collection feed too.

## Constraints

- Python, existing repo style, no new dependencies.
- Do NOT touch `~/babble-live`, do not start or stop any process, do not push.
- Do NOT change the model, tokenizer, checkpoint format, or the training math.
- Pseudonymisation is absolute: no Discord ids, usernames, or raw author identifiers in any feed
  message, stored row, or exported row. Channel ids in consent grant records are fine (they are
  needed to enforce scope and are not personal data), but do not print a person's identity.
- Every piece of collected text that reaches Discord goes through the blocklist withholding and
  `neuter_sample` first. Do not weaken any filtering.
- The feed posts via webhook and must fail soft: a Discord outage or a 4xx must never break
  collection or crash the bot. Log it and move on, exactly as the existing feed does.

## Done means

- Someone pings booper, and a message appears in the collection channel showing the text collected,
  the surface it came from, and the new corpus totals.
- Someone runs the widened opt-in in a channel, and the channel says so; their subsequent messages
  in that channel show up as collected rows.
- Someone withdraws, and the channel reports the withdrawal and how many rows were purged.
- A burst of messages produces one coalesced message, not one per row.
- No training loop starts by itself anywhere.
- The dataset publishes on corpus growth rather than on checkpoints, and the feed says when it did.
- Full test suite passes, with new tests covering: the collection feed events, the coalescing
  window, milestone thresholds at small and large corpus sizes, blocklist/neuter enforcement on
  feed text, and the growth-based publish trigger.

## Checklist
- [x] `CollectionFeed` in `discord_feed.py`: row-added, consent-changed, milestone, publish events; webhook, silent-unconfigured, fail-soft (same infra as `TrainingFeed`, which stays intact).
- [x] Row event shows neutered+blocklist-withheld text, surface label, contributor pseudonym only, running totals (rows / chars / contributors).
- [x] Coalesce rows arriving inside a short window into one message; huge bursts capped (`max_coalesce`), post length capped under Discord's 2000.
- [x] Milestones every N rows and N chars, interval scaling with corpus size (fires now at 54 rows, not spam at 10k); `prime()` stops a restart re-announcing.
- [x] Consent events: default grant, `!babble all` widen, narrow, decline, withdraw with purge counts.
- [x] `CorpusStore.totals()` (rows, chars, contributors) in one scan.
- [x] Wire feed into `core.py` capture + consent paths (pseudonyms only, never raw ids — audited).
- [x] Growth-based HF publisher (`publish.py`): publishes when corpus grows by N rows/chars since last publish, same consent+blocklist gate, persisted baseline (`data/publish_state.json`), announced in feed, fail-soft.
- [x] Wire feed + publisher into `bot.py`; background flush task for the coalescing tail.
- [x] No training loop autostarts anywhere (bot boot only backfills + runs gateway); README/spec/.env wording fixed to make training explicitly opt-in.
- [x] New tests: feed events, coalescing window, milestone thresholds (small + large), blocklist/neuter enforcement on feed text, growth-based publish trigger.
- [x] Full `pytest` suite passes (415 passed); self-reviewed against every "Done means" item.

## Notes
- Collection feed reuses the SAME webhook/channel as the training feed (`BABBLE_LOG_WEBHOOK_URL`) — the ticket wants that channel repurposed, not a second one.
- Trainer's checkpoint-based auto-publish is left intact (manual training still publishes + its tests pass); the NEW growth-based publisher is what keeps the public dataset live during the collection phase, when no trainer runs. This is the additive reading of "re-key to data".
- Row/char milestone intervals scale by magnitude: rows 25→100→500→2500; chars 2k→20k→100k.
