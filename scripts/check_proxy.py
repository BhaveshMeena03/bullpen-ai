"""Does the proxy actually get past YouTube's bot check?

Buying residential bandwidth does not guarantee this works. These pools are
heavily used for scraping, so a share of their IPs are already flagged, and
nobody can tell you in advance which one you will draw. The only honest test
is to fetch real captions through it.

Run this BEFORE trusting the scheduled sync to it:

    export YTDLP_PROXY='http://user:pass@gw.dataimpulse.com:823'
    .venv/bin/python scripts/check_proxy.py

It reports three things, in order of how much they prove:

  1. the exit IP, so you can confirm traffic is actually leaving via the proxy
     rather than your own connection (an unset or malformed variable fails
     open and silently proves nothing)
  2. the channel listing, which is the cheap call
  3. captions for a real video, which is the call that actually gets blocked

A pass on 3 is the only result that means the scheduled sync will work.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Market Bubble #15 -- the video the 14 Aug workflow run was blocked on.
# Testing the one that actually failed beats testing one that might not have.
VIDEO = "NzpJTWJfdg4"


def _ytdlp() -> str:
    local = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "yt-dlp"
    return str(local) if local.exists() else "yt-dlp"


def exit_ip(proxy: str) -> str:
    """Which IP the world sees. Proves the proxy is in the path at all."""
    import urllib.request

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    try:
        with opener.open("https://api.ipify.org", timeout=30) as r:
            return r.read().decode().strip()
    except Exception as e:
        return f"failed: {type(e).__name__}: {str(e)[:90]}"


def main() -> int:
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    if not proxy:
        print("YTDLP_PROXY is not set.\n")
        print("  export YTDLP_PROXY='http://user:pass@gw.dataimpulse.com:823'")
        return 2

    shown = proxy
    if "@" in proxy:                    # never print the password
        scheme, rest = proxy.split("://", 1)
        shown = f"{scheme}://***@{rest.split('@', 1)[1]}"
    print(f"  proxy      : {shown}")

    print("  direct IP  : ", end="", flush=True)
    import urllib.request
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=20) as r:
            direct = r.read().decode().strip()
    except Exception:
        direct = "(unknown)"
    print(direct)

    print("  via proxy  : ", end="", flush=True)
    through = exit_ip(proxy)
    print(through)

    if through == direct:
        print("\n  STOP: traffic is NOT going through the proxy — same IP either")
        print("  way. Check the URL, port and credentials before reading on.")
        return 1
    if through.startswith("failed"):
        print("\n  STOP: could not reach anything through the proxy.")
        return 1

    yt = _ytdlp()
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}

    print("\n  channel listing … ", end="", flush=True)
    listing = subprocess.run(
        [yt, "--flat-playlist", "--print", "%(id)s", "--playlist-end", "3",
         "--proxy", proxy, "https://www.youtube.com/@MarketBubblePod/videos"],
        capture_output=True, text=True, check=False, env=env)
    ids = [x for x in listing.stdout.splitlines() if x.strip()]
    print(f"OK ({len(ids)} videos)" if ids else "FAILED")

    print("  captions      … ", end="", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        got = subprocess.run(
            [yt, "--skip-download", "--write-auto-sub", "--sub-lang", "en",
             "--sub-format", "vtt", "--proxy", proxy,
             "-o", str(Path(tmp) / "%(id)s.%(ext)s"),
             f"https://www.youtube.com/watch?v={VIDEO}"],
            capture_output=True, text=True, check=False, env=env)
        files = list(Path(tmp).glob("*.vtt"))
        if files:
            size = files[0].stat().st_size
            print(f"OK ({size // 1024} KB)")
            print("\n  PASS — this proxy gets past the bot check.")
            print("  Add it as the YTDLP_PROXY secret and the sync will work.")
            return 0

    err = (got.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    print("FAILED")
    print(f"\n  {err[:200]}")
    if "not a bot" in err.lower() or "sign in" in err.lower():
        print("\n  FAIL — this exit IP is flagged. Rotate to a new session and")
        print("  retry; if it keeps failing, the pool is burned for YouTube.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
