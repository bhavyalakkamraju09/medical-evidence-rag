import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "pubmed_20k_rct", "train.txt")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
API_KEY = os.getenv("API_KEY", "changeme")
CACHE_TTL_SECONDS = 600
MAX_CACHE_SIZE = 100
SEMANTIC_THRESHOLD = 0.85
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "embeddings.npy")
