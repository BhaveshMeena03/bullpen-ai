"""Zero-dependency demo server.

Serves the demo page + widget AND fakes the /v1/chat/stream SSE endpoint
with canned answers, so the full chat experience can be demoed without
API keys, Pinecone, or any pip install. Run the real backend (uvicorn
app.main:app) when you have credentials.

    python3 demo/mock_server.py   ->  http://localhost:8000/demo/
"""

import json
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("PORT", 8000))

CANNED = {
    "airdrop": (
        "Here's how the $ANSEM (\"The Black Bull\") airdrop claim works:\n\n"
        "1. Open the $ANSEM Claim tab on Bullpen.\n"
        "2. Connect the wallet linked to your X (Twitter) account — "
        "eligibility is based on your social engagement.\n"
        "3. If you're eligible, your claimable amount from creator fees "
        "shows up automatically. Hit Claim and approve the transaction.\n\n"
        "Heads up: claiming only ever needs a wallet *connection*. Anyone "
        "asking for your seed phrase to \"process your claim\" is a scammer. "
        "(Demo mode — canned answer.)"
    ),
    "ansem_person": (
        "Ansem (@blknoiz06) is a crypto trader and one of the best-known "
        "KOLs in the Solana ecosystem. He's a co-founder of Bullpen and "
        "co-hosts the \"Market Bubble\" podcast with FaZe Banks. The $ANSEM "
        "memecoin — \"The Black Bull\" — is named after him.\n\n"
        "(Demo mode — canned answer.)"
    ),
    "banks": (
        "FaZe Banks is a content creator and co-founder of FaZe Clan, one "
        "of the biggest names in gaming and creator culture. He co-hosts "
        "the \"Market Bubble\" podcast with Ansem — that's the show that "
        "brings a lot of new traders to Bullpen.\n\n"
        "(Demo mode — canned answer.)"
    ),
    "bullpen": (
        "Bullpen is a non-custodial trading terminal that puts three "
        "markets in one place:\n\n"
        "1. Solana spot trading — buy and sell Solana tokens.\n"
        "2. Perps — leveraged futures on assets like BTC, via Hyperliquid.\n"
        "3. Prediction markets — trade real-world event odds, via "
        "Polymarket.\n\n"
        "Non-custodial means you hold your own keys — Bullpen never touches "
        "your funds. It was co-founded by Ansem (@blknoiz06). "
        "(Demo mode — canned answer.)"
    ),
    "ansem_token": (
        "$ANSEM — \"The Black Bull\" — is a memecoin on Solana, named after "
        "Bullpen co-founder Ansem.\n\nWhat makes it different: trading it "
        "generates creator fees, and those fees get airdropped back to the "
        "community via a claim page on Bullpen. Eligibility is based on "
        "your social engagement.\n\nLike all memecoins it's extremely "
        "volatile — I'm a support tool, so I can explain how it works but "
        "can't tell you whether to buy it. (Demo mode — canned answer.)"
    ),
    "perp": (
        "A perp (perpetual future) is a contract that tracks an asset's "
        "price — like BTC — without an expiry date. On Bullpen, perps run "
        "on Hyperliquid.\n\nKey things to know:\n"
        "1. Leverage multiplies both gains and losses.\n"
        "2. If the price moves against you far enough, your position gets "
        "liquidated and you lose your margin.\n"
        "3. Funding payments flow between longs and shorts every 8 hours.\n\n"
        "I'm a support tool, so I can explain the mechanics but can't advise "
        "on positions. (Demo mode — canned answer.)"
    ),
    "wallet": (
        "A crypto wallet is an app that holds the keys to your funds. On "
        "Solana, popular picks are Phantom, Solflare, and Backpack.\n\n"
        "Two things to understand:\n"
        "1. Your public address — safe to share, like an account number.\n"
        "2. Your seed phrase — 12 to 24 words that ARE your money. Never "
        "share it with anyone. Bullpen staff will never ask for it.\n\n"
        "(Demo mode — canned answer.)"
    ),
    "default": (
        "I can walk you through wallets, funding, perps, prediction markets, "
        "the $ANSEM airdrop — or basics like who Ansem and FaZe Banks are "
        "and what Bullpen is. This demo server returns canned answers; run "
        "the real backend with your API keys for live, RAG-grounded "
        "responses. (Demo mode.)"
    ),
}


def pick_answer(question: str) -> str:
    """Cheap keyword routing — the real backend does RAG + Claude instead."""
    q = question.lower()
    if "banks" in q:
        return CANNED["banks"]
    if "airdrop" in q or "claim" in q:
        return CANNED["airdrop"]
    if "$ansem" in q or ("ansem" in q and ("token" in q or "coin" in q or "what is" in q)):
        return CANNED["ansem_token"]
    if "ansem" in q or "who" in q:
        return CANNED["ansem_person"]
    if "bullpen" in q:
        return CANNED["bullpen"]
    if "perp" in q or "leverage" in q or "future" in q or "liquidat" in q:
        return CANNED["perp"]
    if "wallet" in q or "seed" in q or "phrase" in q or "phantom" in q:
        return CANNED["wallet"]
    return CANNED["default"]

SOURCES = [
    {"id": "demo-1", "source_type": "docs", "score": 0.91,
     "metadata": {"title": "Getting Started with Bullpen"}},
    {"id": "demo-2", "source_type": "podcast", "score": 0.84,
     "metadata": {"title": "Market Bubble Ep 12"}},
]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path in ("/", "/demo"):
            self.send_response(302)
            self.send_header("Location", "/demo/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/stream":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        answer = pick_answer(body.get("message", ""))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(event: str | None, data: dict) -> None:
            frame = ""
            if event:
                frame += f"event: {event}\n"
            frame += f"data: {json.dumps(data)}\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()

        emit("sources", SOURCES)
        # Stream the canned answer word by word to mimic token deltas.
        for word in answer.split(" "):
            emit(None, {"text": word + " "})
            time.sleep(0.02)
        emit("done", {})

    def log_message(self, fmt, *args):  # keep demo output quiet
        pass


if __name__ == "__main__":
    print(f"Demo running at http://localhost:{PORT}/demo/")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
