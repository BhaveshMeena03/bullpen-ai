"""Write the four X keys from the environment into .env.

    .venv/bin/python scripts/save_x_keys.py

Run it in the terminal where `read -rs ... && export ...` was used. X shows
each secret exactly once at generation and stores it nowhere retrievable —
not in the console, not over the API — so an exported shell is often the only
surviving copy, and closing that window is what actually loses them.

Nothing is printed but lengths. Existing lines are replaced rather than
appended, so running it twice does not leave duplicates for python-dotenv to
disagree about.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"

KEYS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")


def main() -> int:
    found = {k: os.environ.get(k, "").strip() for k in KEYS}
    missing = [k for k, v in found.items() if not v]
    if missing:
        print("not in this shell: " + ", ".join(missing))
        print("\nThis only works in the terminal where they were exported.")
        print("If that window is gone, they are unrecoverable and have to be")
        print("regenerated at console.x.com — which is harmless: nothing is")
        print("using them yet, and regenerating costs nothing but a re-copy.")
        print("\nRegenerate the Consumer pair FIRST, then the Access Token")
        print("pair, since a new consumer key invalidates tokens made under")
        print("the old one.")
        return 1

    lines = ENV.read_text().splitlines() if ENV.exists() else []
    kept = [ln for ln in lines
            if not any(ln.startswith(f"{k}=") for k in KEYS)]
    while kept and not kept[-1].strip():
        kept.pop()
    kept += ["", "# Written by scripts/save_x_keys.py from the environment."]
    kept += [f"{k}={found[k]}" for k in KEYS]
    ENV.write_text("\n".join(kept) + "\n")
    ENV.chmod(0o600)          # owner-only; it now holds every live secret

    for k in KEYS:
        print(f"  {k:<16} saved ({len(found[k])} chars)")
    print(f"\n  -> {ENV} (permissions 600)")
    print("  verify:  .venv/bin/python scripts/x_whoami.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
