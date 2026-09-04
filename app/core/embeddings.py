"""Sentence embeddings for chunk indexing and query-time retrieval.

Wraps a `sentence_transformers.SentenceTransformer` in a lazy,
process-wide singleton: loading the model costs a download on first use
and a second or two of startup thereafter, so it must not be
instantiated per request or per job.

The configured model (`settings.embedding_model`, default
`all-MiniLM-L6-v2`) produces 384-dimensional vectors, which is what the
`chunks.embedding` pgvector column is declared to hold. Changing the
model to one with a different output size requires a schema change and a
re-index of every chunk.

Embeddings are always L2-normalised, so the cosine distance pgvector
computes with `<=>` is a true cosine distance and `1 - distance` is a
similarity in [-1, 1].
"""

from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the lazily-loaded, process-wide embedding model.

    The first call downloads the weights (~22MB for the default model)
    into the HuggingFace cache and loads them into memory; later calls
    return the same instance.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into normalised vectors.

    Batching matters: encoding N texts in one call is substantially
    faster than N single-text calls, since the model runs them as one
    forward pass.

    Args:
        texts: Texts to embed. An empty list is returned unchanged
            without touching the model.

    Returns:
        One 384-dimensional vector per input text, in the same order,
        each with unit norm.
    """
    if not texts:
        return []
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
