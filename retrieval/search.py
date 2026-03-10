import numpy as np
import faiss

def search_index(index, model, query, chunks, top_k=5, section_filter=None):

    # Encode query
    query_embedding = model.encode([query]).astype("float32")

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search larger candidate pool for filtering
    distances, indices = index.search(query_embedding, 200)

    filtered_results = []

    for score, idx in zip(distances[0], indices[0]):

        chunk_data = chunks[idx]

        if section_filter and chunk_data["section"] != section_filter:
            continue

        filtered_results.append({
            "text": chunk_data["chunk"],
            "section": chunk_data["section"],
            "similarity": float(score)
        })

    # Sort by similarity (higher is better for cosine)
    filtered_results = sorted(
        filtered_results,
        key=lambda x: x["similarity"],
        reverse=True
    )

    return filtered_results[:50]
