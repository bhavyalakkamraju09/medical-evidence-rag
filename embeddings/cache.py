import numpy as np
import os

def save_embeddings(embeddings, path):
    np.save(path, embeddings)

def load_embeddings(path):
    if os.path.exists(path):
        return np.load(path)
    return None
