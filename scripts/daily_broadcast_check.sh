#!/bin/bash
# Ask, the morning after the show, whether Market Bubble posted one
# nobody indexed.
#
# Install (loads a LaunchAgent that runs Friday and Saturday at 11:00):
#
#   scripts/daily_broadcast_check.sh --install
#
# Remove:
#
#   scripts/daily_broadcast_check.sh --uninstall
#
# It notifies and stops there. Transcription runs locally and takes about
# twenty minutes, and the summary it produces goes out under the account's
# name, so the part that needs a person still gets one. This only removes
# the part that was easy to forget: noticing.
#
# Friday and Saturday at 11:00, from the show's own posting record: all
# nine live broadcasts went up on a Thursday, between 20:31 and 20:39 UTC
# — Friday 02:01-02:09 IST — and ran three to four hours, ending around
# 06:00 IST. By 11:00 Friday the stream is long over and X has had time
# to swap the live feed for the full recording, which is what the six
# hour settle window in find_new_broadcasts waits for.
#
# Saturday is the retry, for the week the laptop is shut on Friday. A
# daily schedule was the first version and it was mostly waste: six of
# every seven runs could only ever say "nothing new", and each one still
# costs a tenth of a dollar in X reads.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.mbubblesearch.broadcastcheck"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

notify() {
  # Terminal-notifier is nicer but is not installed by default, and a
  # dependency that has to be remembered defeats the point of this.
  osascript -e "display notification \"$2\" with title \"$1\"" 2>/dev/null || true
}

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
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>5</integer>
          <key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>6</integer>
          <key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$ROOT/.broadcast-check.log</string>
  <key>StandardErrorPath</key><string>$ROOT/.broadcast-check.log</string>
</dict>
</plist>
PLISTEOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "  installed — runs Friday and Saturday at 11:00"
    echo "  log: $ROOT/.broadcast-check.log"
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
echo "── $(date '+%Y-%m-%d %H:%M') ──"

# Exit 1 means something is missing, which is the interesting case, so the
# check must not take the script down with it under `set -e`.
OUTPUT="$(.venv/bin/python scripts/find_new_broadcasts.py --limit 30 2>&1)" \
  && STATUS=0 || STATUS=$?
echo "$OUTPUT"

if [ "$STATUS" -eq 1 ]; then
  notify "Market Bubble" "A broadcast is not indexed yet — run add_broadcast.py"
elif [ "$STATUS" -gt 1 ]; then
  # A silent failure here looks exactly like "nothing new" forever.
  notify "Market Bubble" "Broadcast check failed — see .broadcast-check.log"
fi
exit 0
