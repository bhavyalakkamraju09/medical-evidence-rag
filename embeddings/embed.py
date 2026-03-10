from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def generate_embeddings(model, chunks):
    texts = [c["chunk"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    return np.array(embeddings).astype("float32")
