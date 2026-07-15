"""Shared embedding helper: batches large inputs and retries rate limits.

Voyage free-tier keys are limited to 3 requests/min and 10K tokens/min, so
bulk ingestion must chunk its calls and back off politely. Paid keys sail
through the same code path with fewer, larger batches.
"""

import asyncio
import logging

import voyageai
from voyageai import error as voyage_error

logger = logging.getLogger(__name__)

# ~24K chars ≈ 6-7K tokens — safely under the free tier's 10K TPM per call.
BATCH_CHAR_BUDGET = 24_000
MAX_RETRIES = 6


def _batches(texts: list[str], char_budget: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and size + len(text) > char_budget:
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


async def embed_texts(
    client: voyageai.AsyncClient,
    texts: list[str],
    *,
    model: str,
    dimension: int,
    input_type: str,
) -> list[list[float]]:
    """Embed `texts`, transparently batching and retrying 429s with backoff."""
    embeddings: list[list[float]] = []
    batches = _batches(texts, BATCH_CHAR_BUDGET)
    for i, batch in enumerate(batches):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await client.embed(
                    texts=batch,
                    model=model,
                    input_type=input_type,
                    output_dimension=dimension,
                )
                embeddings.extend(result.embeddings)
                break
            except voyage_error.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                wait = min(25 * attempt, 90)
                logger.info(
                    "Voyage rate limit (batch %d/%d, try %d) — waiting %ds",
                    i + 1, len(batches), attempt, wait,
                )
                await asyncio.sleep(wait)
        if len(batches) > 1 and i < len(batches) - 1:
            await asyncio.sleep(21)  # free tier: 3 requests/minute
    return embeddings
