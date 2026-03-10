import os
import time
import json
import numpy as np
import redis
import faiss

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# Prevent FAISS OpenMP crash
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------------
# Project Imports
# ----------------------------
from ingestion.load_data import load_pubmed_data
from ingestion.preprocess import preprocess_dataframe
from ingestion.chunking import chunk_dataframe

from embeddings.embed import load_embedding_model
from embeddings.build_index import build_faiss_index
from embeddings.cache import load_embeddings

from retrieval.search import search_index
from retrieval.rerank import rerank

from generation.prompt_builder import build_rag_prompt
from generation.generator import generate_answer

from evaluation.verifier import verify_answer

import config

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Medical RAG API")

# ----------------------------
# Security (Proper Bearer Auth)
# ----------------------------
security = HTTPBearer()

def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    if credentials.credentials != config.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ----------------------------
# Redis Client (Docker-Compatible)
# ----------------------------
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
    decode_responses=True
)

# ----------------------------
# Request Schema
# ----------------------------
class QueryRequest(BaseModel):
    query: str


# ----------------------------
# Global System Objects
# ----------------------------
df = None
chunks = None
model = None
index = None


# ----------------------------
# Startup Event
# ----------------------------
@app.on_event("startup")
def load_system():
    global df, chunks, model, index

    print("Loading system...")

    df = load_pubmed_data(config.DATA_PATH)
    df = preprocess_dataframe(df)

    chunks = chunk_dataframe(
        df,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP
    )

    model = load_embedding_model(config.EMBEDDING_MODEL)

    embeddings = load_embeddings(config.EMBEDDINGS_PATH)

    if embeddings is None:
        raise RuntimeError("Embeddings not found. Run run_pipeline.py first.")

    index = build_faiss_index(embeddings)

    print("System ready.")


# ----------------------------
# Health Endpoint
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Main RAG Endpoint (Protected)
# ----------------------------
@app.post("/ask")
def ask_question(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):

    try:
        total_start = time.time()
        query = request.query

        # -----------------------------------
        # 1️⃣ Encode Query
        # -----------------------------------
        query_embedding = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        # -----------------------------------
        # 2️⃣ Check Redis Semantic Cache
        # -----------------------------------
        for key in redis_client.scan_iter("semantic_cache:*"):

            entry_raw = redis_client.get(key)
            if not entry_raw:
                continue

            entry = json.loads(entry_raw)
            cached_embedding = np.array(entry["embedding"]).reshape(1, -1)

            similarity = cosine_similarity(
                query_embedding,
                cached_embedding
            )[0][0]

            if similarity > config.SEMANTIC_THRESHOLD:

                cached_response = entry["response"]

                cached_response["metrics"]["cached"] = True
                cached_response["metrics"]["semantic_similarity"] = round(float(similarity), 3)
                cached_response["metrics"]["retrieval_seconds"] = 0
                cached_response["metrics"]["generation_seconds"] = 0
                cached_response["metrics"]["total_seconds"] = round(time.time() - total_start, 3)

                return cached_response

        # -----------------------------------
        # 3️⃣ Retrieval
        # -----------------------------------
        retrieval_start = time.time()

        initial_results = search_index(
            index=index,
            model=model,
            query=query,
            chunks=chunks,
            top_k=30,
            section_filter="conclusions"
        )

        results = rerank(query, initial_results, top_k=5)
        results = [r for r in results if r["rerank_score"] > 0]
        results = results[:3]

        retrieval_latency = time.time() - retrieval_start

        # -----------------------------------
        # 4️⃣ Generation
        # -----------------------------------
        generation_start = time.time()

        prompt = build_rag_prompt(query, results)
        answer = generate_answer(prompt)
        verification = verify_answer(answer, results)

        generation_latency = time.time() - generation_start
        total_latency = time.time() - total_start

        response = {
            "query": query,
            "answer": answer,
            "evidence": results,
            "verification": verification,
            "metrics": {
                "retrieval_seconds": round(retrieval_latency, 3),
                "generation_seconds": round(generation_latency, 3),
                "total_seconds": round(total_latency, 3),
                "cached": False
            }
        }

        # -----------------------------------
        # 5️⃣ Store In Redis With TTL
        # -----------------------------------
        cache_key = f"semantic_cache:{hash(query)}"

        redis_client.setex(
            cache_key,
            config.CACHE_TTL_SECONDS,
            json.dumps({
                "embedding": query_embedding.tolist(),
                "response": response
            })
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))