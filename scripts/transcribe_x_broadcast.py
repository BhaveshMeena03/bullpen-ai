"""Transcribe an X live broadcast into the Episode JSON the search ingests.

    export GROQ_API_KEY=...          # free key from console.groq.com
    .venv/bin/python scripts/transcribe_x_broadcast.py URL [URL ...]

Why this exists: roughly half the show is not on YouTube. The guest
interviews go out live and stay on X, and the full "LIVE W/ ..." broadcasts
are posted there too. A viewer asked the search about the Squire founder,
got nothing, and reasonably concluded the tool was broken — it was not, it
simply could not see that half of the catalogue.

X serves a caption track for uploaded clips, and none at all for live
broadcasts. The video is downloadable either way, so the missing piece is
only the words, which is what this fills in.

Design notes worth knowing before changing anything:

  audio only    The video is never needed and is ~40x larger. Audio is
                pulled straight to 16 kHz mono, which is what Whisper
                resamples to anyway — sending anything richer costs upload
                time and buys nothing.
  time chunks   Split by duration, not by byte count, so each chunk's
                offset into the broadcast is known exactly and the stitched
                timestamps stay true. A byte-based split cannot tell you
                where it landed.
  overlap       Consecutive chunks share a few seconds, and segments
                falling inside the seam are dropped from the later chunk.
                Cutting mid-word otherwise loses or duplicates a phrase at
                every boundary — invisible in a spot check, and exactly the
                sort of thing that makes one quote in fifty subtly wrong.
  pacing        The free tier allows 7,200 audio-seconds per hour, so a
                five-hour broadcast takes a few hours of wall clock. It
                waits rather than failing, and says how long it is waiting.

Costs nothing on the free tier. On paid it is $0.111 per hour of audio for
whisper-large-v3 — about 56 cents for a five-hour show.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.captions import collapse_repeats  # noqa: E402
from app.dedupe import SAME_RECORDING  # noqa: E402
from app.episode_store import merge as merge_episodes  # noqa: E402
from scripts.fetch_x_episodes import (  # noqa: E402
    OUT,
    overlap_with_existing,
    probe,
    title_from,
)

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Local transcription, and the default. Measured on an M3 Pro against a
# two-minute sample, model already cached:
#
#   whisper-large-v3-turbo    7.9s   -> 15x realtime   ~20 min for a 5h show
#   whisper-large-v3         47.0s   -> 2.6x realtime  ~2 hours
#
# Turbo carries a slightly higher word error rate (12% against 10.3%), and
# on the sample the two produced the same text. It is also comfortably
# better than the YouTube auto-captions this index already trusts for 24
# episodes — those have no punctuation at all. Six times the speed for
# quality that is still above the existing bar is the right default; pass
# --model large-v3 when a particular broadcast is worth the wait.
MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# large-v3 rather than the turbo variant: 10.3% word error rate against 12%.
# Both are free within the same limits, and this tool's entire claim is that
# it quotes people accurately, so the more accurate model is the only
# defensible default.
MODEL = "whisper-large-v3"

# Minutes of audio per request. 20 minutes at 16 kHz mono MP3 is about 5 MB,
# comfortably inside the 25 MB free-tier ceiling with room for a dense
# stretch of speech to come out larger than average.
CHUNK_MINUTES = 20
OVERLAP_SECONDS = 4

_STATUS = re.compile(r"(?:twitter|x)\.com/([^/]+)/status/(\d+)")


def _bin(name: str) -> str:
    local = ROOT / ".venv" / "bin" / name
    return str(local) if local.exists() else (shutil.which(name) or name)


def download_audio(url: str, dest: Path) -> Path:
    """Pull the broadcast's audio as 16 kHz mono MP3.

    Format selection matters more than it looks. An X broadcast replay has
    NO audio-only rendition — all four streams are muxed video+audio — so
    "bestaudio/best" silently falls through to `best` and downloads the
    highest-bitrate video of a five-hour show. Measured: 838 MB and still
    climbing at 0.8 MB/s, heading for several gigabytes, to extract audio
    that ends up around 70 MB.

    `worstvideo+bestaudio/worst` takes the smallest rendition instead. The
    video track is discarded seconds later, and Whisper resamples to 16 kHz
    mono regardless, so nothing about the transcript is worse for it.
    """
    print("  downloading audio (smallest rendition; video is discarded)…")
    # Sixteen fragments at once, then one at a time if that fails.
    #
    # A five-hour broadcast arrives as ~9,600 HLS fragments, and yt-dlp
    # fetches them one at a time by default. Each is a separate HTTPS
    # request of roughly 150KB, so the download is bounded by round-trip
    # latency and not by bandwidth at all: measured 120 KB/s on a link that
    # does 14.9 MB/s, under 1% of it. Sixteen at once turns hours into
    # minutes and costs nothing but sockets.
    #
    # On some streams they race each other, though: "No such file or
    # directory: ...mp4.part-Frag68", reproducibly, on two of thirty-three
    # episodes re-downloaded in one sitting. Serial is slow and always
    # worked, so it is the retry rather than the default — and a broadcast
    # that fails to download here has failed the whole pipeline, twenty
    # minutes before anybody finds out.
    for fragments in ("16", "1"):
        # Leftover fragment state makes the retry resume a download that no
        # longer exists, and fail differently the second time.
        for stale in dest.parent.glob(f"{dest.name}.*"):
            if stale.suffix != ".mp3":
                stale.unlink(missing_ok=True)
        try:
            subprocess.run(
                [_bin("yt-dlp"), "--socket-timeout", "30", "--no-warnings",
                 "--concurrent-fragments", fragments,
                 "-f", "worstaudio/worstvideo+bestaudio/worst",
                 "--extract-audio", "--audio-format", "mp3",
                 "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -b:a 32k",
                 "-o", str(dest.with_suffix(".%(ext)s")), url],
                check=True, timeout=7200,
            )
            break
        except subprocess.CalledProcessError:
            if fragments == "16":
                print("  fragments raced — retrying one at a time (slower)")
                continue
            raise
    got = dest.with_suffix(".mp3")
    if not got.exists():
        raise SystemExit(f"audio download produced no file for {url}")
    return got


def duration_of(path: Path) -> float:
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def slice_audio(src: Path, start: float, length: float, dest: Path) -> Path:
    subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-y", "-ss", str(start), "-t",
         str(length), "-i", str(src), "-ac", "1", "-ar", "16000",
         "-b:a", "32k", str(dest)],
        check=True, timeout=1800,
    )
    return dest


def transcribe_local(path: Path, model: str) -> list[dict]:
    """One audio file -> [{start, text}], entirely on this machine.

    No API, no key, no rate limit — which is the whole point. Groq's free
    tier meters two hours of audio per hour, and its paid tier was
    unavailable to sign up for, so anything that depends on it is
    rationed by someone else's capacity.
    """
    import mlx_whisper  # imported lazily: only this path needs it
    result = mlx_whisper.transcribe(
        str(path), path_or_hf_repo=MLX_MODELS[model], language="en",
    )
    return [{"start": float(s["start"]), "text": s["text"].strip()}
            for s in result.get("segments", []) if s.get("text", "").strip()]


def transcribe_chunk(path: Path, api_key: str) -> list[dict]:
    """One chunk -> [{start, text}], retrying past the rate limiter.

    A 429 here is expected rather than exceptional: the free tier's hourly
    audio allowance is smaller than a long broadcast, so the correct
    behaviour is to wait for the window to move, not to fail the run.
    """
    for attempt in range(8):
        with open(path, "rb") as fh:
            try:
                resp = httpx.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (path.name, fh, "audio/mpeg")},
                    data={"model": MODEL, "response_format": "verbose_json",
                          "language": "en", "temperature": "0"},
                    timeout=900,
                )
            except httpx.HTTPError as exc:
                print(f"      network error ({exc}); retrying")
                time.sleep(20 * (attempt + 1))
                continue
        if resp.status_code == 429:
            wait = float(resp.headers.get("retry-after") or 0) or 90
            print(f"      rate limited, waiting {wait:.0f}s "
                  f"(free tier allows 2h of audio per hour)")
            time.sleep(wait + 5)
            continue
        if resp.status_code >= 400:
            raise SystemExit(f"Groq error {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return [{"start": float(s["start"]), "text": s["text"].strip()}
                for s in body.get("segments", []) if s.get("text", "").strip()]
    raise SystemExit("gave up after repeated rate limiting")


def run_transcriber(path: Path, opts) -> list[dict]:
    """Local by default; Groq only when explicitly asked for."""
    if opts.engine == "groq":
        return transcribe_chunk(path, opts.api_key)
    return transcribe_local(path, opts.model)


def sample_overlap(audio: Path, opts, existing: list[dict]) -> tuple[float, str]:
    """Transcribe three short samples and estimate how much is already indexed.

    Worth its own pass because the free tier meters audio, not requests: a
    five-hour broadcast costs about two and a half hours of waiting, and the
    guard may well reject it at the end for duplicating a YouTube episode
    that was uploaded from the same show. Six minutes of sampling answers
    that before the wait rather than after it.

    Samples are taken from the start, middle and end. A broadcast that
    matches an upload matches it throughout; one that only shares an
    introduction does not.
    """
    total = duration_of(audio)
    picks = [total * 0.10, total * 0.50, total * 0.85]
    segs: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, start in enumerate(picks):
            piece = slice_audio(audio, start, 120.0,
                                Path(tmp) / f"probe{i}.mp3")
            got = run_transcriber(piece, opts)
            segs.extend({"t": start + g["start"], "text": g["text"]}
                        for g in got)
    if len(segs) < 20:
        return 0.0, ""
    return overlap_with_existing(segs, existing)


def transcribe(audio: Path, opts) -> list[dict]:
    total = duration_of(audio)
    chunk = CHUNK_MINUTES * 60
    n = max(1, math.ceil(total / chunk))
    print(f"  {total/60:.0f} min of audio -> {n} chunk(s) of "
          f"{CHUNK_MINUTES} min")

    segments: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            start = max(0.0, i * chunk - (OVERLAP_SECONDS if i else 0))
            length = min(chunk + OVERLAP_SECONDS, total - start)
            if length <= 0.5:
                break
            piece = slice_audio(audio, start, length,
                                Path(tmp) / f"part{i:03d}.mp3")
            size_mb = piece.stat().st_size / 1_048_576
            print(f"    chunk {i+1}/{n}  {start/60:5.1f}–"
                  f"{(start+length)/60:5.1f} min  {size_mb:4.1f} MB")
            got = run_transcriber(piece, opts)
            # Shift into broadcast time, and drop anything inside the seam
            # so the overlap does not appear twice.
            floor = segments[-1]["t"] if segments else -1.0
            for s in got:
                t = start + s["start"]
                if t <= floor:
                    continue
                segments.append({"t": round(t, 2), "text": s["text"]})
            print(f"      {len(got):5d} segments, running total "
                  f"{len(segments)}")
    # Whisper repetition loops, removed before anything downstream sees them.
    cleaned = collapse_repeats(segments)
    if len(cleaned) < len(segments):
        print(f"    dropped {len(segments) - len(cleaned)} repeated segments")
    return cleaned


def _groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        # Fall back to the project's .env so this behaves like everything
        # else here rather than demanding a differently-shaped setup.
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("GROQ_API_KEY is not set. Either drop --engine groq to run "
                 "locally, or get a free key at console.groq.com and put it "
                 "in .env as GROQ_API_KEY=...")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="x.com status URLs (live broadcasts)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--date", metavar="YYYY-MM-DD",
                    help="the date this was RECORDED, when it differs from "
                         "when it was posted. A live broadcast is posted the "
                         "day it airs, so its post date is right. A clip is "
                         "posted whenever someone got around to cutting it, "
                         "and taking that as the air date makes an old "
                         "conversation look like the newest thing in the "
                         "index — which matters, because the answers reason "
                         "about recency ('what does he think now').")
    ap.add_argument("--max-overlap", type=float, default=SAME_RECORDING)
    ap.add_argument("--engine", choices=["local", "groq"], default="local",
                    help="local runs on this machine with no rate limit "
                         "(default); groq needs GROQ_API_KEY")
    ap.add_argument("--keep-audio", action="store_true",
                    help="save the audio to audio/<episode_id>.mp3. Speaker "
                         "labelling needs it, and re-downloading an X "
                         "broadcast costs 900MB a second time.")
    ap.add_argument("--model", choices=sorted(MLX_MODELS), default="turbo",
                    help="local model: turbo is ~6x faster, large-v3 is "
                         "slightly more accurate")
    args = ap.parse_args()
    args.api_key = ""

    if args.engine == "groq":
        args.api_key = _groq_key()
    opts = args

    print(f"  engine: {args.engine}"
          + (f" ({args.model}, ~15x realtime)" if args.engine == "local" else ""))

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    by_id = {e["episode_id"]: e for e in existing}
    # Only what THIS run transcribed gets written back. The rest of by_id is
    # a snapshot from before a transcription that may have run for hours, and
    # writing it back is exactly how another process's work used to vanish.
    written_ids: set[str] = set()
    print(f"  {len(existing)} episode(s) already on file\n")

    for url in args.urls:
        m = _STATUS.search(url)
        if not m:
            print(f"  SKIP {url}: not an x.com/<user>/status/<id> URL")
            continue
        handle, status_id = m.groups()
        episode_id = f"x-{status_id}"

        meta = probe(url)
        if meta is None:
            continue
        print(f"\n  {title_from(meta)[:64]}")

        others = [e for e in by_id.values() if e["episode_id"] != episode_id]
        with tempfile.TemporaryDirectory() as tmp:
            audio = download_audio(url, Path(tmp) / "audio")

            # Six minutes of sampling before committing to hours of it.
            print("  sampling 3x2min to check it is not already indexed…")
            ratio, where = sample_overlap(audio, opts, others)
            if ratio >= args.max_overlap:
                print(f"  SKIP  sample says ~{ratio:.0%} of this is already "
                      f"in {where}")
                print("        (pass --max-overlap 1.0 to transcribe anyway)")
                continue
            print(f"  sample overlap {ratio:.0%} — transcribing in full")
            segments = transcribe(audio, opts)

            # Keep the audio when asked. Who is speaking is a fact about
            # the sound and cannot be recovered from the words, so an
            # episode transcribed and discarded has to be downloaded again
            # — 900MB again, for an X broadcast — before it can be
            # labelled. The pipeline wants both, so it takes both once.
            if args.keep_audio:
                keep_dir = ROOT / "audio"
                keep_dir.mkdir(exist_ok=True)
                kept = keep_dir / f"{episode_id}.mp3"
                shutil.copy2(audio, kept)
                print(f"  kept the audio at {kept.relative_to(ROOT)}")

        if len(segments) < 20:
            print(f"  SKIP {url}: only {len(segments)} segments, looks empty")
            continue

        ratio, where = overlap_with_existing(segments, others)
        if ratio >= args.max_overlap:
            print(f"  SKIP  {ratio:.0%} of this is already in {where}")
            print("        (pass --max-overlap 1.0 to add it anyway)")
            continue
        if ratio > 0.15:
            print(f"  note  {ratio:.0%} of this also appears in {where}")

        ts = meta.get("timestamp")
        written_ids.add(episode_id)
        by_id[episode_id] = {
            "episode_id": episode_id,
            "title": title_from(meta),
            "url": f"https://x.com/{handle}/status/{status_id}",
            # X seeks on ?t=<seconds> but rejects the trailing "s" that
            # YouTube requires, and the deep-link builder keys the suffix
            # off this field. Citations land on the moment either way.
            "platform": "other",
            "published_at": args.date or (
                datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                if ts else None),
            "segments": segments,
        }
        mins = segments[-1]["t"] / 60
        print(f"  added {len(segments)} segments, {mins:.0f} min")

    # A transcription runs for many minutes; the file it read at startup is
    # stale by the time it finishes. Merging under a lock, re-reading inside
    # it, is what keeps a concurrent fetch from being silently reverted.
    new = [e for e in by_id.values() if e["episode_id"] in written_ids]
    merge_episodes(new, out_path)
    total = len(json.loads(out_path.read_text()))
    print(f"\n{total} episodes total -> {out_path.relative_to(ROOT)}")
    print("Now re-ingest:  .venv/bin/python scripts/ingest_episodes.py")


if __name__ == "__main__":
    main()
