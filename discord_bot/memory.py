"""Short-lived conversation memory for the concierge bot.

The web page replays the whole conversation on every request, so the concierge
can ask a clarifying question and use the answer. The Discord bot sent no
history at all, which broke the bot's most useful habit: its deposit answer
ends "once you answer those two questions I can point you at the exact next
step", and there was no way to answer them.

Deliberately forgetful:

* Keyed on (user, channel) so a conversation in one channel can't bleed into
  another, and two people asking at once never see each other's context.
* Expires after a few minutes of silence. Someone returning an hour later has
  moved on, and stale context makes answers worse, not better.
* Keeps only the last few turns. History is resent on every request, so
  unbounded history means an unbounded bill.
* Bounded number of conversations, oldest evicted, so a big server can't grow
  this without limit.

Nothing is written to disk. Restarting the bot forgets everything, which is
the right default for support chat that may mention someone's balance.
"""

from __future__ import annotations

TTL_SECONDS = 600.0      # 10 minutes of silence ends the conversation
MAX_TURNS = 6            # three exchanges; history is resent every request
MAX_CONVERSATIONS = 5_000


class ConversationMemory:
    def __init__(
        self,
        *,
        ttl: float = TTL_SECONDS,
        max_turns: int = MAX_TURNS,
        max_conversations: int = MAX_CONVERSATIONS,
    ):
        self._ttl = ttl
        self._max_turns = max_turns
        self._max_conversations = max_conversations
        # key -> (last_used_monotonic, [{"role": ..., "content": ...}, ...])
        self._store: dict[tuple[int, int], tuple[float, list[dict]]] = {}

    def get(self, user_id: int, channel_id: int, now: float) -> list[dict]:
        entry = self._store.get((user_id, channel_id))
        if not entry:
            return []
        last, turns = entry
        if now - last > self._ttl:
            del self._store[(user_id, channel_id)]
            return []
        return list(turns)

    def append(
        self,
        user_id: int,
        channel_id: int,
        question: str,
        answer: str,
        now: float,
    ) -> None:
        key = (user_id, channel_id)
        _, turns = self._store.get(key, (now, []))
        turns = turns + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self._store[key] = (now, turns[-self._max_turns:])
        self._evict(now)

    def forget(self, user_id: int, channel_id: int) -> bool:
        return self._store.pop((user_id, channel_id), None) is not None

    def _evict(self, now: float) -> None:
        # Drop anything already expired first; only fall back to evicting the
        # least recently used if the map is still over the cap.
        for key in [k for k, (last, _) in self._store.items()
                    if now - last > self._ttl]:
            del self._store[key]
        while len(self._store) > self._max_conversations:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
