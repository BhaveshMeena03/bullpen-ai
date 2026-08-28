#!/bin/bash
# Start watching for Thursday night's show, and index it when it ends.
#
# Install (a LaunchAgent that starts the watcher Friday 04:00 local time):
#
#   scripts/daily_broadcast_check.sh --install
#
# Remove:
#
#   scripts/daily_broadcast_check.sh --uninstall
#
# Run it by hand any time:
#
#   .venv/bin/python scripts/watch_for_broadcast.py --dry-run
#
# 04:00 Friday, from the show's own record: all nine live broadcasts went
# up on a Thursday between 20:31 and 20:39 UTC — Friday 02:01-02:09 IST —
# and ran three to four hours. So the earliest one can end is about 05:00
# IST, and starting an hour before that means the watcher is already
# running when the recording appears rather than discovering it late.
#
# It indexes. It does not post. The summary reaches the site and anyone
# who asks; nothing goes out under the account's name unattended.
#
# THIS ONLY RUNS IF THE MAC IS AWAKE. Transcription is local — that is
# what the twenty minutes is. A sleeping laptop indexes nothing, and the
# watcher will simply start whenever the machine next wakes and catch the
# show a few hours late. To have it happen at 06:00 while you sleep, the
# Mac has to be plugged in and scheduled to be awake; `pmset` does that
# and needs your password, so it is yours to run, not this script\'s:
#
#   sudo pmset repeat wake F 03:55:00
#
# The watcher itself holds sleep off with caffeinate once it starts a
# transcription, so a Mac that is awake at 04:00 will stay awake through
# the pipeline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.mbubblesearch.broadcastwatch"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

case "${1:-}" in
  --install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/scripts/daily_broadcast_check.sh</string>
    <string>--watch</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>5</integer>
    <key>Hour</key><integer>4</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>$ROOT/.broadcast-check.log</string>
  <key>StandardErrorPath</key><string>$ROOT/.broadcast-check.log</string>
</dict>
</plist>
PLISTEOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "  installed — starts watching Friday 04:00, indexes when the show ends"
    echo "  log: $ROOT/.broadcast-check.log"
    echo
    echo "  It cannot run while the Mac is asleep. To wake it first:"
    echo "    sudo pmset repeat wake F 03:55:00"
    exit 0
    ;;
  --uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "  removed"
    exit 0
    ;;
esac

cd "$ROOT"
echo "== $(date "+%Y-%m-%d %H:%M %Z") =="
# caffeinate -i: the watcher polls for hours and then transcribes, and a
# Mac that sleeps partway leaves a half-written episode behind.
exec caffeinate -i .venv/bin/python -u scripts/watch_for_broadcast.py
