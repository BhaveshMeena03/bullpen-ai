"""Pull real Market Bubble captions from YouTube and convert them into the
Episode JSON the podcast search ingests.

Uses yt-dlp to download auto-generated VTT subtitles (no video download),
parses each caption cue into a {t, text} segment, dedupes YouTube's rolling
auto-caption overlap, and writes data/episodes.json.

    .venv/bin/python scripts/fetch_episodes.py VIDEO_ID [VIDEO_ID ...]
    .venv/bin/python scripts/fetch_episodes.py --latest 5   # newest N MB eps

If YouTube blocks the automated download (bot check / 429), either pass your
browser cookies:

    .venv/bin/python scripts/fetch_episodes.py --cookies chrome VIDEO_ID

or download the .vtt yourself (browser extension / online tool) and convert
the local files — no network needed:

    .venv/bin/python scripts/fetch_episodes.py --vtt ep9.vtt=VIDEO_ID ...

Then ingest:
    curl -X POST localhost:8100/v1/podcast/ingest \\
      -H 'content-type: application/json' -d @data/episodes.json
"""

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
def _ytdlp() -> str:
    """Prefer the project venv, fall back to PATH.

    The venv path is right on a laptop and wrong everywhere else — a CI
    runner pip-installs yt-dlp onto PATH and has no .venv, which is what
    broke the first scheduled sync.
    """
    local = ROOT / ".venv" / "bin" / "yt-dlp"
    if local.exists():
        return str(local)
    return shutil.which("yt-dlp") or "yt-dlp"


YTDLP = _ytdlp()


def _proxy_args() -> list[str]:
    """--proxy flags for yt-dlp, or nothing when YTDLP_PROXY is unset.

    Read at call time rather than import time so a test or a shell can set it
    without reimporting the module.
    """
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    return ["--proxy", proxy] if proxy else []
# Parsing moved to app/captions.py so the server-side clipper uses the
# exact same cue handling the index was built from.
from app.captions import _clean, coalesce, parse_vtt  # noqa: E402

CHANNEL = "https://www.youtube.com/@MarketBubble/videos"
OUT = ROOT / "data" / "episodes.json"

def from_vtt_file(path: Path, video_id: str, title: str | None = None) -> dict | None:
    """Convert a local .vtt caption file into an Episode (no network)."""
    raw = parse_vtt(path.read_text(encoding="utf-8", errors="ignore"))
    segments = coalesce(raw)
    if not segments:
        print(f"  ! empty transcript in {path}")
        return None
    print(f"  ✓ {video_id}: {len(segments)} segments — {title or path.name}")
    return {
        "episode_id": video_id,
        "title": title or path.stem,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "platform": "youtube",
        "segments": segments,
    }



# yt-dlp's failure modes look the same from the outside but need opposite
# responses: a blocked IP needs credentials or a different host, a video
# with no subtitles yet just needs waiting. Naming which one it is turns a
# silent weekly stall into something actionable.
_BLOCK_SIGNS = (
    "sign in to confirm", "not a bot", "confirm you're not a bot",
    "http error 403", "http error 429", "too many requests",
    "unable to extract", "player response", "failed to extract",
)


def _diagnose(stderr: str) -> str:
    err = (stderr or "").lower()
    last = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    tail = last[-1][:160] if last else "(no stderr)"
    if any(sign in err for sign in _BLOCK_SIGNS):
        return f"BLOCKED by YouTube (bot check / rate limit) — {tail}"
    if "no subtitles" in err or "no automatic captions" in err:
        return f"no captions published yet — {tail}"
    return tail


def fetch(
    video_id: str, cookies_browser: str | None = None, attempts: int = 3
) -> dict | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    # YouTube gates caption downloads behind a JS challenge + bot check.
    # --remote-components pulls yt-dlp's solver; --cookies-from-browser and a
    # JS runtime (deno, on PATH) get past the bot check. See README.
    cookie_args = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    # A residential proxy is what makes this work anywhere but a laptop. From a
    # datacenter IP -- GitHub Actions, Fly, any VPS -- YouTube answers "Sign in
    # to confirm you're not a bot" for the video page AND the captions
    # endpoint, both measured. So the scheduled sync could detect a new episode
    # and never be able to ingest it, which is exactly what happened on 14 Aug.
    #
    # Optional on purpose: unset, everything behaves as before, so a local run
    # needs no configuration and no secret.
    proxy_args = _proxy_args()
    common = ["--remote-components", "ejs:github", *cookie_args, *proxy_args]
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    title = video_id
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Single invocation: downloads captions AND prints the title, so we
            # only hit YouTube once per episode (two calls trips rate limits).
            proc = subprocess.run(
                [YTDLP, "--skip-download", "--write-auto-sub", "--write-sub",
                 "--sub-lang", "en", "--sub-format", "vtt", *common,
                 # --print alone implies simulate (nothing downloads!);
                 # --no-simulate makes it print the title AND write files.
                 # Title and airdate on one line. The date is what lets an
                 # answer say "in May he argued X; by July he'd shifted" —
                 # without it every chunk is timeless. Still one call per
                 # episode; two trips YouTube's rate limit.
                 "--print", "%(title)s\t%(upload_date)s", "--no-simulate",
                 "-o", str(tmp_path / "%(id)s.%(ext)s"), url],
                capture_output=True, text=True, check=False, env=env,
            )
            first = (proc.stdout.strip().splitlines() or [title])[0]
            title, _, raw_date = first.partition("\t")
            title = title or video_id
            # yt-dlp gives YYYYMMDD, or "NA" when it has no date.
            published_at = (f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                            if len(raw_date) == 8 and raw_date.isdigit()
                            else None)
            vtts = list(tmp_path.glob("*.vtt"))
            if vtts:
                segments = coalesce(
                    parse_vtt(vtts[0].read_text(encoding="utf-8", errors="ignore"))
                )
                if segments:
                    print(f"  ✓ {video_id}: {len(segments)} segments — {title}")
                    return {
                        "episode_id": video_id, "title": title, "url": url,
                        "platform": "youtube", "published_at": published_at,
                        "segments": segments,
                    }
        why = _diagnose(proc.stderr)
        if attempt < attempts:
            wait = 20 * attempt  # back off — YouTube throttles rapid pulls
            print(f"  … {video_id} failed (try {attempt}/{attempts}): {why} "
                  f"— waiting {wait}s")
            time.sleep(wait)
    print(f"  ! {video_id} failed after {attempts} tries: {_diagnose(proc.stderr)}")
    return None


def latest_ids(n: int, cookies_browser: str | None = None) -> list[str]:
    cookie_args = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    out = subprocess.run(
        # Deliberately NOT proxied. The 14 Aug CI run listed the channel
        # fine and printed "1 new episode(s): ['NzpJTWJfdg4']" before the
        # per-video fetch was refused -- so the bot check guards the video,
        # not the listing. Since the listing runs daily and a fetch only when
        # an episode appears, proxying it would spend roughly 80% of the
        # bandwidth on the one call that does not need it.
        [YTDLP, "--flat-playlist", "--print", "%(id)s",
         "--remote-components", "ejs:github", *cookie_args, CHANNEL],
        capture_output=True, text=True, check=False, env=env,
    )
    ids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()][:n]
    if not ids:
        err = (out.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  ! channel listing failed — {err[:140]}")
    return ids


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    # Local-file mode: --vtt path=VIDEO_ID [path=VIDEO_ID ...]
    if argv[0] == "--vtt":
        episodes = []
        for spec in argv[1:]:
            path_str, _, vid = spec.partition("=")
            ep = from_vtt_file(Path(path_str), vid or Path(path_str).stem)
            if ep:
                episodes.append(ep)
        OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
        total = sum(len(e["segments"]) for e in episodes)
        print(f"\nWrote {len(episodes)} episodes ({total} segments) -> {OUT}")
        return 0

    cookies = None
    if argv and argv[0] == "--cookies":
        cookies, argv = argv[1], argv[2:]

    if argv and argv[0] == "--latest":
        ids = latest_ids(int(argv[1]) if len(argv) > 1 else 3, cookies)
    else:
        ids = argv

    # Resume support: keep episodes already fetched in a previous run.
    episodes: list[dict] = []
    if OUT.exists():
        try:
            episodes = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            episodes = []
    done = {e["episode_id"] for e in episodes}
    ids = [v for v in ids if v not in done]

    print(f"Fetching {len(ids)} episode(s) ({len(done)} already cached)...")
    for i, vid in enumerate(ids):
        ep = fetch(vid, cookies)
        if ep:
            episodes.append(ep)
            # Incremental save — a long run that dies keeps its progress.
            OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
        if i < len(ids) - 1:
            time.sleep(12)  # be polite; rapid pulls trip YouTube throttling

    OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
    total = sum(len(e["segments"]) for e in episodes)
    print(f"\nWrote {len(episodes)} episodes ({total} segments) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
