"""The hundred questions the rerank comparison is scored on.

Kept apart from the harness because the questions are the experiment. A
comparison is only worth the set it runs on, and a set of softballs would
have told us the change was free either way.

Six kinds, each measuring something different:

  PROBES      the answer is in the archive and the episode is known, so
              "found it" means the right episode rather than a confident
              paragraph about the wrong one. Mined from the transcripts,
              not invented -- every one was read before it was written.

  ABSENT      the answer is NOT in the archive. Verified by searching
              every transcript -- as a substring, and for every
              spelling the captions actually produce. The first version
              of this list checked "mt gox" and "friend.tech" on a word
              boundary; the transcripts say "Mount Gox" and "Frentech",
              so three correct answers were scored as inventions. Refusing
              is the correct behaviour, and a setting that starts
              answering these is hallucinating, which is the whole worry
              about widening the pool. This is the category that matters.

  HOSTS       "what did X say about Y", where the risk is not a missing
              answer but a line credited to the wrong man. This is where
              the $254,000 reply came from, and where the one
              misattribution in the last run showed up.

  GUESTS      the archive is half guests, and a question about Jesse or
              Perk has to survive the same speaker filter the hosts do.

  LOGGED      what people actually tagged the account with, typos and
              all. Nobody types "what did ansem say about hyperliquid".

  MESSY       the shapes a mention really arrives in -- a greeting, a
              summons, a question buried after somebody else's handle.
"""

from __future__ import annotations

# (question, expected_episode_id or None)
# The episode is the one the answer lives in, read out of the transcript
# before the question was written.
PROBES = [
    ("who was the guy who sold all his eth holding", "F4OhqZjtVkY"),
    ("which bankless guy sold his ethereum", "F4OhqZjtVkY"),
    ("who capitulated on eth after arguing with ansem for years",
     "F4OhqZjtVkY"),
    ("how much did bonk go from and to in market cap",
     "x-2072780197179666519"),
    ("how much got liquidated in shorts on that bitcoin move",
     "x-2090537938899845434"),
    ("how much did ansem give away in a week", "Ff1shGhhQzQ"),
    ("how much was wired to tristan thompson", "wksdB2BgW3g"),
    ("how much did he burn in chicago setting up the entity",
     "wksdB2BgW3g"),
    ("how much did saylor lose on that bitcoin sale", "liVvayvxoRU"),
    ("how much leverage did hyperliquid let bullpen offer after the close",
     "x-2070244072342827487"),
    ("how much of the token have they bought back and burned",
     "x-2077854740575092752"),
    ("what did they say about h100 prices a year ago", "F4OhqZjtVkY"),
    ("how much is micron up", "lTWv-SIEFpo"),
    ("how much is anthropic guaranteed to spend on cloud services",
     "VRkszcJSgWU"),
    ("what did they say robinhood is worth", "TYX2FuacIhE"),
    ("what was the treasury tvl on hyperliquid", "qFeglFI5bac"),
]

# Searched every transcript for these before adding them. Refusing is
# right; answering is the failure.
ABSENT = [
    "what did they say about the bybit hack",
    "what did ansem say about lazarus group",
    "what did they say about ftx repayments",
    "what did they say about olympus dao",
    "what did banks say about buying a football club",
    "what did ansem say about his tax bill",
    "what did they say about tether being investigated",
    "what did they say about dogecoin's founder",
    "what did they say about interest rate cuts in december",
    "what did they say about the wormhole hack",
    "what did they say about do kwon",
    "what did banks say about shiba inu",
    "what did ansem say about polkadot",
    "what did they say about el salvador buying bitcoin",
    "what did ansem say about sushiswap",
    "what did banks say about safemoon",
    "what did they say about the ripple lawsuit",
    "what did ansem say about chainlink staking",
    "what did they say about the poly network hack",
    "what did banks say about harmony one",
]

HOSTS = [
    "what did ansem say about hyperliquid",
    "what did ansem say about ethereum",
    "what did ansem say about solana",
    "what did ansem say about bonk",
    "what did ansem say about pump fun",
    "what did ansem say about zcash",
    "what did ansem say about memecoins",
    "what did ansem say about airdrops",
    "what did ansem say about his solana price target",
    "what did ansem say about the ansem token",
    "what did ansem say about trading psychology",
    "what did ansem say about bitcoin dominance",
    "what did banks say about streaming",
    "what did banks say about solana",
    "what did banks say about his portfolio",
    "what did banks say about hyperliquid",
    "what did banks say about content",
    "what did banks say about faze",
    "what did banks say about kick",
    "what did banks say about the ansem token",
    "what did banks say about polymarket",
    "what did banks say about gambling",
    "what did ansem and banks disagree about",
    "what did they say about prediction markets",
    "what did they say about ai versus crypto",
]

GUESTS = [
    "what did jesse pollak say about base",
    "what did brian armstrong say about suing the sec",
    "what did kendrick perkins say about relationships",
    "what did luca netz say about pudgy penguins",
    "what did tjr say about attention",
    "what did mert say about solana",
    "what did mizkif say about giving his chat money",
    "what did tushar jain say about intelligence",
    "what did orangie say about being a fortnite pro",
    "what did austin federa say about solana",
    "what did mike dudas say about crypto media",
    "what did the helius ceo say about solana",
    "what did the venice ai ceo say about the dollar",
    "what did mayne say about selling to kraken",
    "what did lucas bruder say about jito",
]

LOGGED = [
    "what did ansem say about women and crypto",
    "what did ansem say about bonk",
    "Ansem saying, like, we need to onboard girls.",
    "what did ansem and banks say about $ansem",
    "who was the guy who sold all his eth holding",
    "create tg heavily bundled",
    "crazy search engine technology",
    "yoo banks you really need to see this what did banks say about "
    "hyperliquid",
    "what did they say about robotics",
    "why does ansem think ethereum is done",
    "how did he turn 500 dollars into 40 million",
    "blackrock told an nba player not to buy bitcoin",
]

MESSY = [
    "hey what was that thing ansem said about the trojan horse",
    "yo @mbubbleSearch did banks ever talk about his own portfolio",
    "someone said solana was going to 150 who was it",
    "the episode where they talk about gta 6",
    "which episode had the coinbase guy",
    "what was the ferrari thing they were laughing about",
    "did anyone on the show call the eth bottom",
    "who said trading is becoming the new celebrity culture",
    "what's the deal with the z500 leaderboard",
    "they mentioned a guy who made 40 million on trump who was that",
    "was there anything about north korea",
    "what did they say about pudgy penguins and women in crypto",
    "someone bought a car for 25 million as a joke right",
    "who on the show is most bearish on eth",
]


def all_questions() -> list[tuple[str, str | None, str]]:
    """(question, expected_episode, category) with duplicates removed.

    The probes carry ground truth and must survive deduplication against
    the logged set, which contains the same question with none. Losing
    that quietly is exactly how the first run of this comparison
    reported a tie between two settings that differ.
    """
    rows: list[tuple[str, str | None, str]] = []
    rows += [(q, ep, "probe") for q, ep in PROBES]
    rows += [(q, None, "absent") for q in ABSENT]
    rows += [(q, None, "host") for q in HOSTS]
    rows += [(q, None, "guest") for q in GUESTS]
    rows += [(q, None, "logged") for q in LOGGED]
    rows += [(q, None, "messy") for q in MESSY]

    keep: dict[str, tuple[str, str | None, str]] = {}
    for q, ep, kind in rows:
        low = q.lower().strip()
        if low not in keep or (ep and not keep[low][1]):
            keep[low] = (q, ep, kind)
    return list(keep.values())


if __name__ == "__main__":
    from collections import Counter
    rows = all_questions()
    counts = Counter(kind for _q, _e, kind in rows)
    print(f"\n  {len(rows)} questions")
    for kind, n in counts.most_common():
        print(f"    {kind:8} {n}")
    print()
