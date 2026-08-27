"""The Dockerfile and .dockerignore have to agree.

This is here because they disagreed, and nothing caught it. A COPY was
added for a file inside a directory .dockerignore excluded. Docker does
not warn and carry on — the build fails outright — and on Render a failed
build means the previous container keeps serving. So the URL stayed up,
/healthz stayed green, and the bot stopped answering for twenty minutes
with no signal anywhere that a deploy had failed.

A build would catch it, but a build needs Docker running and takes
minutes. This reads the two files and takes milliseconds, so it runs on
every commit instead of on the ones where someone remembered.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def copy_sources(dockerfile: str) -> list[str]:
    """The host paths a Dockerfile copies in.

    Skips `COPY --from=`, which reads from an earlier build stage rather
    than the build context and so is not subject to .dockerignore.
    """
    sources: list[str] = []
    # Line continuations first, so a wrapped COPY is read as one line.
    for line in re.sub(r"\\\n", " ", dockerfile).splitlines():
        line = line.strip()
        if not re.match(r"(?i)^(COPY|ADD)\s", line):
            continue
        parts = line.split()[1:]
        if any(p.startswith("--from=") for p in parts):
            continue
        parts = [p for p in parts if not p.startswith("--")]
        # The last token is the destination inside the image.
        sources.extend(parts[:-1])
    return sources


def patterns(dockerignore: str) -> list[str]:
    return [ln.strip() for ln in dockerignore.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def is_excluded(path: str, rules: list[str]) -> bool:
    """Whether Docker would leave `path` out of the build context.

    Docker evaluates every rule and the LAST one that matches decides, so
    `data/` followed by `!data/highlights.json` excludes the directory and
    then puts that one file back. Evaluating in order and keeping the last
    verdict is what makes that work; stopping at the first match would
    report the file as excluded when Docker would include it.
    """
    verdict = False
    for rule in rules:
        negated = rule.startswith("!")
        pattern = rule.lstrip("!").rstrip("/")
        if (path == pattern
                or fnmatch.fnmatch(path, pattern)
                or path.startswith(pattern + "/")
                or fnmatch.fnmatch(path, pattern + "/*")):
            verdict = not negated
    return verdict


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="no Dockerfile")
def test_every_copied_path_survives_dockerignore():
    """Every COPY source must be in the build context.

    If this fails, the image does not build at all — so it is a deploy
    outage, not a missing feature.
    """
    rules = patterns(DOCKERIGNORE.read_text())
    for source in copy_sources(DOCKERFILE.read_text()):
        assert not is_excluded(source, rules), (
            f"Dockerfile copies {source!r}, but .dockerignore excludes it. "
            f"The build will fail with 'not found'. Add '!{source}' to "
            f".dockerignore, below the rule that excludes it."
        )


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="no Dockerfile")
def test_every_copied_path_exists_on_disk():
    """A COPY of a path that is not in the repo fails the same way."""
    for source in copy_sources(DOCKERFILE.read_text()):
        if any(ch in source for ch in "*?["):
            continue
        assert (ROOT / source).exists(), (
            f"Dockerfile copies {source!r}, which does not exist")


def test_the_matcher_understands_docker_last_match_wins():
    """Guards the check itself.

    A matcher that stopped at the first match would call highlights.json
    excluded and fail a build that Docker would run happily — and one that
    ignored directory prefixes would pass the build that actually broke
    production. Both directions are pinned here.
    """
    rules = ["data/", "!data/highlights.json"]
    assert is_excluded("data/episodes.json", rules)
    assert not is_excluded("data/highlights.json", rules), "last match wins"
    assert is_excluded("data", rules)

    # Without the exception — the state that took the bot down.
    assert is_excluded("data/highlights.json", ["data/"])

    # Unrelated paths are untouched.
    assert not is_excluded("app", ["data/", "tests/"])
    assert is_excluded("tests", ["data/", "tests/"])


def test_the_highlight_pool_is_shipped():
    """The specific file whose absence produced silent wrong behaviour.

    Missing, the bot answers a compliment with nothing at all — the one
    failure that looks to a reader like the account is broken rather than
    thinking.
    """
    sources = copy_sources(DOCKERFILE.read_text())
    assert "data/highlights.json" in sources, (
        "the Dockerfile no longer ships the highlight pool")
    assert not is_excluded("data/highlights.json",
                           patterns(DOCKERIGNORE.read_text()))
