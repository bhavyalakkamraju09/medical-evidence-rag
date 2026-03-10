import faiss
import numpy as np

def build_faiss_index(embeddings, nlist=100):

    dimension = embeddings.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Quantizer
    quantizer = faiss.IndexFlatIP(dimension)

    # IVF index
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # Train
    index.train(embeddings)

    # Add embeddings
    index.add(embeddings)

    # Accuracy vs speed tradeoff
    index.nprobe = 10

    return index