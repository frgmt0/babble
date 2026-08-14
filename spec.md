# Booper ignores some pings silently: log the drops, catch role mentions
> run: run-20260814-booper-ignores-some-pings-silently-log-t · branch: beckett/run-booper-ignores-some-pings-silently-log-t · created: 2026-08-14T06:23:44.435Z

## Goal
ro (user 1151230208783945818) reported: "some users are reporting pinging the bot and then being unable to get a response back at all. can you investigate. it seems to be just some users even after !babble accept".

This is the kowo-co/babble repo (the babble/booper Discord bot). I did the first pass of the investigation on the live box; here is what I found, so don't redo it:

- Zero `bot.error` events in the log other than one `missing_token` from before the token was set. Nothing is throwing.
- Every single `bot.ping` in the log got a matching `bot.sent`. There is no case of a message reaching the handler and being answered with silence.
- The bot is now in **2 guilds** (it got shared), but the log only ever shows 3 distinct channel hashes, all from the original server. Not one message from the second guild has ever reached the handler.
- `babble/core.py` `handle_message` has this gate:

      if not (msg.mentions_bot or msg.reply_to_is_bot):
          return []

  It returns an empty list and logs **nothing at all**. Every message the bot decides not to answer vanishes without a trace, which is exactly why I cannot tell a permissions problem from a mention-detection problem from a user who typed the bot's name without actually mentioning it.

- `mentions_bot` is computed in `babble/bot.py` as `bool(self.user and self.user in message.mentions)`. `message.mentions` does not include **role mentions**, so a user who pings a role the bot has (very common in other servers) is silently ignored. The bot also runs on `discord.Intents.default()` plus `message_content`; the members intent is off.

The job, in priority order:

1. **Make the silent drop observable.** Log every message the bot chooses not to act on, with the reason — not a mention, no remembered exchange, author is a bot, whatever the branch was — plus the channel and guild (pseudonymised the same way every other log line is: use the existing `Pseudonymiser` / `log.user()` / `log.channel()` helpers, never raw ids or usernames, and respect the existing consent rules about logging content — log the FACT of the drop and its reason, not the message text, when there is no consent). Add a guild field to the relevant log events so a multi-server deployment is debuggable at all; pseudonymise it like the rest.
2. **Log guild + channel visibility at startup.** On ready, log every guild the bot is in and, for each, how many text channels it can actually see and send in, so a permissions gap in a shared server is visible from the log instead of requiring a manual API poke.
3. **Handle role mentions.** Treat a mention of a role the bot has as a ping, the same as a direct user mention. Keep ignoring `@everyone`/`@here` — those must NOT trigger a response.
4. **Log when a reply/send fails for permissions.** A `discord.Forbidden` on reply already funnels into `bot.error`; make sure the reason is distinguishable (missing send perms vs anything else) and includes the pseudonymised channel, so "the bot can see it but can't answer" is a one-line log search.

Constraints:
- Python, existing dependencies only. Do NOT add new dependencies.
- **Never log a raw user id, username, guild name, or raw channel id.** Everything identifying goes through the existing pseudonymiser. This is a consent-sensitive project and the logs are something I read routinely — that guarantee cannot regress.
- Do not log message content for users who have not consented. The existing `log.preview(..., allowed=...)` pattern is the model to follow.
- Do not change the consent flow itself, the blocklist/content filter, the HuggingFace export, or the training loop.
- Do not make the bot respond to messages that are not directed at it. Widening the trigger to role mentions is in scope; responding to every message in a channel is emphatically not.
- Keep the "a bad message must never kill the gateway" guarantee in `_think` intact.

Done means: the full test suite is green (it is at 159 now, all must still pass), there are new tests covering the drop-reason logging, the role-mention trigger, and that `@everyone`/`@here` still does not trigger a response; and after this lands I can read the log and tell, for any user who says the bot ignored them, whether the message reached the bot at all and why it was not answered.

## Checklist
- [x] Add `guild_id` to `IncomingMessage`/`ReactionEvent`, `guild()` pseudonymiser on `Pseudonymiser`/`EventLog`/`NullLog`
- [x] Log `bot.dropped` (reason, pseudonymised user/channel/guild, char count, never text) for every silent `return []` branch in `handle_message` (author-is-bot, not-addressed)
- [x] Thread `guild` field through existing log events that already carry `channel` (`bot.ping`, `bot.generate`, `consent.prompt` x2, `command`, `reaction.ignored`)
- [x] On `on_ready`, log a `bot.guild` event per guild with text-channel count / visible count / sendable count (`_guild_visibility` helper)
- [x] Treat a mention of a role the bot holds as a ping (`_mentions_bot` helper in `bot.py`), while excluding the guild's default (`@everyone`) role so `@everyone`/`@here` never trigger a response
- [x] Distinguish `bot.error` reason on reply failure: `forbidden` (403) vs `http_error` (anything else), each carrying the pseudonymised channel + guild (`_send_reply` helper)
- [x] New tests: role-mention trigger, `@everyone`/`@here` non-trigger, guild-visibility counting, Forbidden-vs-other logging (`tests/test_bot.py`), drop-reason logging + guild field on/off (`tests/test_core.py`)
- [x] Full test suite green: 178/178 (159 original + 19 new), no new dependencies added
- [x] README "Watching it" section documents the new `bot.dropped` / `bot.error` reasons / `bot.guild` events

## Notes
- "no remembered exchange" (mentioned in the ticket as an example drop reason) does not currently correspond to a silent-drop code path: when a reply-to-bot message has no matching exchange, `handle_message` already falls through to `_respond()` and answers normally rather than dropping, so there was nothing to add logging to there. The analogous case that *does* silently drop — a 👍 reaction on an unremembered message — was already logged as `reaction.ignored`; it now also carries `channel`/`guild`.
- Extracted `_send_reply` (bot.py) and kept `_log_drop` (core.py) as small named helpers, matching the existing `_log_skip`/`_log_blocked` pattern, so the Forbidden/HTTPException distinction is unit-testable without a live gateway connection.
