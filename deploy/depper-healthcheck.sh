#!/bin/sh
# depper [health].command for booper (see depper.toml and docs/DEPPER.md).
#
#   sh deploy/depper-healthcheck.sh [UNIT]      (default: depper-babble.service)
#
# Healthy means: the unit is active, and babble.jsonl contains a `bot.ready`
# event whose timestamp is at or after the moment systemd started the unit's
# current main process. That is the same proof deploy/update-live.sh used
# ("a fresh bot.ready after the restart"), but keyed to the process start
# time instead of a byte offset, so it also works for `depper health` long
# after the deploy, and a bot.ready left in the log by an earlier process
# (the old babble-bot unit, or the previous release) can never satisfy it.
#
# depper runs this on the host from the current release directory with the
# [env] table exported (secrets are not), so BABBLE_LOG_DIR comes from
# depper.toml. Read-only: it runs `systemctl --user show`, grep and date.
# Exit 0 = healthy; anything else = not (yet) healthy, reason on stderr.
set -eu

unit=${1:-depper-babble.service}
log_dir=${BABBLE_LOG_DIR:-}
if [ -z "$log_dir" ]; then
  echo "BABBLE_LOG_DIR is not set (it comes from [env] in depper.toml)" >&2
  exit 1
fi

active=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null || true)
if [ "$active" != "active" ]; then
  echo "$unit is '${active:-unknown}', not active" >&2
  exit 1
fi

pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)
case $pid in
  '' | 0 | *[!0-9]*)
    echo "$unit has no main process (MainPID='$pid')" >&2
    exit 1
    ;;
esac

# --timestamp=unix prints "@<seconds>"; it needs systemd >= 248.
started=$(systemctl --user show "$unit" --timestamp=unix -p ExecMainStartTimestamp --value 2>/dev/null || true)
started=${started#@}
case $started in
  '' | *[!0-9]*)
    echo "cannot read the start time of $unit (got '$started')" >&2
    exit 1
    ;;
esac

# The newest bot.ready. babble.jsonl is append-only and rotates by size to
# babble.jsonl.1, so look in the live file first and fall back to the most
# recent rotated one in case the rotation happened right after startup.
# Records are json.dumps() output: {"ts": "<iso8601>", "component": "bot",
# "event": "bot.ready", ...}.
ready_ts=""
for f in "$log_dir/babble.jsonl" "$log_dir/babble.jsonl.1"; do
  [ -f "$f" ] || continue
  line=$(grep -F '"event": "bot.ready"' "$f" 2>/dev/null | tail -n 1 || true)
  if [ -n "$line" ]; then
    ready_ts=$(printf '%s\n' "$line" | sed -n 's/^{"ts": "\([^"]*\)".*/\1/p')
    break
  fi
done

if [ -z "$ready_ts" ]; then
  echo "no bot.ready in $log_dir/babble.jsonl yet ($unit pid $pid started @$started)" >&2
  exit 1
fi

ready_epoch=$(date -d "$ready_ts" +%s 2>/dev/null || true)
case $ready_epoch in
  '' | *[!0-9]*)
    echo "cannot parse bot.ready timestamp '$ready_ts'" >&2
    exit 1
    ;;
esac

if [ "$ready_epoch" -lt "$started" ]; then
  echo "newest bot.ready ($ready_ts) predates $unit pid $pid start (@$started); waiting for this process to reach ready" >&2
  exit 1
fi

echo "bot.ready at $ready_ts; $unit pid $pid started @$started"
