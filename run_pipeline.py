# ===============================================
# FULL RAG PIPELINE (ML ENGINEER VERSION)
# ===============================================

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent FAISS OpenMP crash

# ---------- Core Imports ----------
from ingestion.load_data import load_pubmed_data
from ingestion.preprocess import preprocess_dataframe
from ingestion.chunking import chunk_dataframe

from embeddings.embed import load_embedding_model, generate_embeddings
from embeddings.build_index import build_faiss_index
from embeddings.cache import save_embeddings, load_embeddings

from retrieval.search import search_index
from retrieval.rerank import rerank

from evaluation.metrics import precision_at_k
from evaluation.metrics_extended import recall_at_k, hit_rate, mrr

from generation.prompt_builder import build_rag_prompt
from generation.generator import generate_answer

import config


# ===============================================
# MAIN PIPELINE
# ===============================================
def main():

    # ------------------------------------------
    # 1. Load Data
    # ------------------------------------------
    print("Loading data...\n")
    df = load_pubmed_data(config.DATA_PATH)

    print("Section Distribution:")
    print(df["section"].value_counts())
    print()

    # ------------------------------------------
    # 2. Preprocess
    # ------------------------------------------
    df = preprocess_dataframe(df)

    # ------------------------------------------
    # 3. Chunking
    # ------------------------------------------
    print("Chunking...\n")
    chunks = chunk_dataframe(
        df,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP
    )

    print(f"Total Chunks: {len(chunks)}\n")

    # ------------------------------------------
    # 4. Load Embedding Model
    # ------------------------------------------
    model = load_embedding_model(config.EMBEDDING_MODEL)

    # ------------------------------------------
    # 5. Load or Generate Embeddings
    # ------------------------------------------
    embeddings = load_embeddings(config.EMBEDDINGS_PATH)

    if embeddings is None:
        print("Generating embeddings...\n")
        embeddings = generate_embeddings(model, chunks)
        save_embeddings(embeddings, config.EMBEDDINGS_PATH)
    else:
        print("Loaded cached embeddings.\n")

    print("Embeddings shape:", embeddings.shape)
    print("Embeddings dtype:", embeddings.dtype)
    print()

    # ------------------------------------------
    # 6. Build FAISS Index
    # ------------------------------------------
    print("Building FAISS index...\n")
    index = build_faiss_index(embeddings)

    # ------------------------------------------
    # 7. Query
    # ------------------------------------------
    query = "What treatments are effective for Parkinson disease?"

    # ---- Initial Retrieval (Top 50) ----
    initial_results = search_index(
        index=index,
        model=model,
        query=query,
        chunks=chunks,
        top_k=50,
        section_filter="conclusions"
    )

    # ---- Cross-Encoder Re-ranking ----
    results = rerank(query, initial_results, top_k=5)

    # ------------------------------------------
    # 8. Display Final Top-5 Evidence
    # ------------------------------------------
    print("Top 5 Re-ranked Evidence:\n")

    for i, r in enumerate(results):
        print(f"[{i+1}]")
        print("Section:", r["section"])
        print("Similarity:", round(r["similarity"], 4))
        print("Rerank Score:", round(r["rerank_score"], 4))
        print(r["text"])
        print()

    # ------------------------------------------
    # 9. Retrieval Evaluation Metrics
    # ------------------------------------------
    keywords = ["treatment", "therapy", "levodopa", "dopamine", "intervention"]

    print("Evaluation Metrics:")
    print("Precision@K:", precision_at_k(results, keywords))
    print("Recall@K:", recall_at_k(results, keywords))
    print("Hit Rate:", hit_rate(results, keywords))
    print("MRR:", mrr(results, keywords))
    print()

    # ------------------------------------------
    # 10. Build RAG Prompt
    # ------------------------------------------
    prompt = build_rag_prompt(query, results)

    print("Generating Answer...\n")

    # ------------------------------------------
    # 11. Generate Final Answer (Local LLM)
    # ------------------------------------------
    answer = generate_answer(prompt)

    print("Final Answer:\n")
    print(answer)

    print("\n===============================================")


# ===============================================
# ENTRY POINT
# ===============================================
if __name__ == "__main__":
    main()