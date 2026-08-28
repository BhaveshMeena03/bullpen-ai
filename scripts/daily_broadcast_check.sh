#!/bin/bash
# Ask once a day whether Market Bubble posted a show nobody indexed.
#
# Install (loads a LaunchAgent that runs at 11:00 every day):
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
# 11:00 rather than overnight because the show runs late and the finder
# ignores anything posted in the last six hours — a stream that ended at
# 2am is ready by mid-morning, not at 6am.

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
  <dict><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$ROOT/.broadcast-check.log</string>
  <key>StandardErrorPath</key><string>$ROOT/.broadcast-check.log</string>
</dict>
</plist>
PLISTEOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "  installed — runs at 11:00 daily"
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
