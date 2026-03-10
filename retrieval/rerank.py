from sentence_transformers import CrossEncoder

# Load once (global)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, top_k=5):
    """
    Re-rank retrieved chunks using cross-encoder.
    """

    pairs = [(query, c["text"]) for c in candidates]

    scores = cross_encoder.predict(pairs)

    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    # Sort descending (higher = better)
    ranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]
