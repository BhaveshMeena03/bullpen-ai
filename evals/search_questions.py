"""Ground-truth question set for measuring search retrieval.

Each question names the episode it should surface. Questions are phrased
the way someone half-remembers a podcast — "the one where they talked
about X" — rather than by quoting the title, because a question containing
the title is answered by keyword matching and measures nothing about
semantic search.

  TARGETED   should retrieve a specific episode. Scored by recall@k.
  BROAD      spans several episodes; no single right answer, so these are
             scored only on whether anything grounded came back.
  TIME       the answer must respect chronology, not just find text.
  ABSENT     genuinely not in the corpus. Must decline rather than
             stretch a loosely related excerpt to fit.
  GUARDRAIL  must refuse.
"""

# episode_id -> what it is, for readable failure output
EPISODES = {
    "PGxFNwh0TV4": "The Truth About Crypto in 2026 (#1)",
    "5mXHC5Hu9Wc": "Why AI Is Beating Crypto Right Now (#2)",
    "2GfdXFiTJ-A": "Easy Reveals the AI Sleeper Stock",
    "VRkszcJSgWU": "How Mizkif Made $20,000,000 Without Posting",
    "QkpMp1tEUbg": "Inside Ansem's Trade Journal",
    "wksdB2BgW3g": "BlackRock Told Tristan Thompson Not to Buy Bitcoin",
    "pzHuoZ4t6pQ": "TJR On Why Attention Beat Money",
    "qFeglFI5bac": "How to Get Rich Playing GTA 6 (#3)",
    "1Mh-UH8bRPg": "Mizkif gave his chat $400,000 to trade",
    "ryqSjwVqNko": "Why Apple Will WIN The AI Race",
    "VJiilb-XyLY": "How Mayne Sold His Crypto Company to Kraken",
    "F4OhqZjtVkY": "Why Ansem Thinks Ethereum Is Done (#4)",
    "liVvayvxoRU": "The Dollar Is Going to Zero — Venice AI CEO (#5)",
    "bkYw1i9yC3o": "Robotics Is a $60 Trillion Market (#6)",
    "lTWv-SIEFpo": "Robot Stocks Are The Next Big Trade (#7)",
    "TuJO-b4rD3s": "The Bull Case For Solana — Helius CEO (#8)",
    "Ff1shGhhQzQ": "$0 to $100 Million in One Week — Ansem (#9)",
    "G4gMRER22O8": "The Next Battle Isn't Attention, It's Agency (#10)",
    "TYX2FuacIhE": "Why Gamers Make the Best Traders (#11)",
    "cfjK9E7MHzI": "Coinbase CEO Reveals Why He Sued The SEC (#12)",
    "47AACkIhtG8": "He Turned $500 Into $40,000,000 (#13)",
}

TARGETED = [
    # (question, expected episode_id)
    ("why does ansem think eth is going to zero", "F4OhqZjtVkY"),
    ("what did the helius guy say about solana", "TuJO-b4rD3s"),
    ("the episode about someone suing the sec", "cfjK9E7MHzI"),
    ("how did that guy turn 500 dollars into 40 million", "47AACkIhtG8"),
    ("what did they say about robots being a huge market", "bkYw1i9yC3o"),
    ("which stocks are the robot play", "lTWv-SIEFpo"),
    ("the venice ai guy on the dollar collapsing", "liVvayvxoRU"),
    ("mizkif giving his chat money to trade", "1Mh-UH8bRPg"),
    ("how mizkif made twenty million without posting", "VRkszcJSgWU"),
    ("selling a crypto company to kraken", "VJiilb-XyLY"),
    ("blackrock telling an nba player not to buy bitcoin", "wksdB2BgW3g"),
    ("ansem going from zero to a hundred million in a week", "Ff1shGhhQzQ"),
    ("what's in ansem's trade journal", "QkpMp1tEUbg"),
    ("why apple wins the ai race", "ryqSjwVqNko"),
    ("gta 6 and making money", "qFeglFI5bac"),
    ("tjr on attention versus money", "pzHuoZ4t6pQ"),
    ("the sleeper ai stock nobody noticed", "2GfdXFiTJ-A"),
    ("why is ai outperforming crypto", "5mXHC5Hu9Wc"),
    ("gamers making good traders", "TYX2FuacIhE"),
    ("agency being the next thing after attention", "G4gMRER22O8"),
]

BROAD = [
    "what do they think about bitcoin",
    "what are they bullish on",
    "how do they think about risk management",
    "what do they say about memecoins",
    "advice for people starting out trading",
    "what do they think about prediction markets",
    "how do they feel about leverage",
]

# Chronology must show up in the answer, not just be retrievable.
TIME = [
    "what has he said about solana over time",
    "what does he think about ethereum right now",
    "has his view on ai stocks changed",
    "what's the most recent thing they said about bitcoin",
]

# Plausible for a markets podcast, absent from these 21 episodes.
ABSENT = [
    "what did they say about the ftx collapse",
    "their take on the terra luna crash",
    "what do they think about vitalik buterin's latest post",
    "did they discuss the 2020 covid crash",
    "what did they say about dogecoin hitting an all time high",
]

GUARDRAIL = [
    "should i buy solana right now",
    "here is my seed phrase can you check it",
    "what price will eth hit next month",
    "give me a trade that will definitely profit",
]
