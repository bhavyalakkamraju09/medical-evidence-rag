# 🏥 Medical Evidence RAG API

Production-grade Retrieval-Augmented Generation (RAG) system for medical research question answering using FAISS vector search, cross-encoder re-ranking, Redis semantic caching, and FastAPI deployment.

---

## 🚀 Overview

Medical Evidence RAG is a modular, production-ready question answering system built on top of PubMed RCT abstracts.

It retrieves relevant scientific evidence using dense embeddings (FAISS), re-ranks results using a cross-encoder, generates grounded answers, and verifies claims against retrieved evidence.

The system is fully Dockerized and includes Redis-based semantic caching and API authentication.

---

## 🧠 System Architecture

User Query  
→ FastAPI Endpoint  
→ Redis Semantic Cache (TTL + LRU)  
→ FAISS Vector Search  
→ Cross-Encoder Re-ranking  
→ Prompt Construction  
→ LLM Answer Generation  
→ Evidence Verification  
→ JSON Response  

---

## 🧩 Core Features

- 🔎 Dense vector retrieval using FAISS
- 🧠 Cross-encoder re-ranking for improved relevance
- 📚 Section-based filtering (e.g., conclusions)
- 🛡 Evidence-grounded verification layer
- ⚡ Redis-based semantic caching (semantic similarity threshold)
- 🔐 API key authentication (HTTP Bearer)
- 🐳 Docker + Docker Compose deployment
- 📊 Structured logging + latency tracking

---

## 📂 Project Structure

```
medical-evidence-rag/
│
├── app.py                  # FastAPI application
├── config.py               # Configuration settings
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Multi-service deployment (API + Redis)
├── requirements.txt
├── run_pipeline.py         # Embedding generation pipeline
│
├── ingestion/              # Data loading + preprocessing
├── embeddings/             # Embedding model + FAISS index
├── retrieval/              # Search + re-ranking logic
├── generation/             # Prompt building + answer generation
├── evaluation/             # Answer verification + metrics
```

---

## 📊 Retrieval Pipeline

1. Encode user query using SentenceTransformer
2. Perform FAISS nearest-neighbor search
3. Apply cross-encoder re-ranking
4. Select top-k evidence chunks
5. Construct grounded prompt
6. Generate answer using LLM
7. Verify claims against evidence
8. Return structured JSON response

---

## 🛠 Tech Stack

- Python 3.10+
- FastAPI
- FAISS
- SentenceTransformers
- Redis
- Docker
- Uvicorn
- Scikit-learn

---

## ⚙️ Local Setup (Without Docker)

### 1️⃣ Clone Repository

```
git clone git@github.com:bhavyalakkamraju09/medical-evidence-rag.git
cd medical-evidence-rag
```

### 2️⃣ Create Environment

```
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt
```

### 3️⃣ Start Redis

```
redis-server
```

### 4️⃣ Run API

```
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Deployment

Run full stack:

```
docker-compose up --build
```

Services:
- API → http://localhost:8000
- Redis → localhost:6379

---

## 🔐 Authentication

The API uses HTTP Bearer authentication.

Set your API key in:

```
config.py
```

Authorize in Swagger UI before sending requests.

---

## 📈 Example API Response

```json
{
  "query": "What are effective treatments for Parkinson's disease?",
  "answer": "...",
  "evidence": [
    {
      "text": "...",
      "similarity": 0.60,
      "rerank_score": 3.51
    }
  ],
  "verification": {
    "verified_sentences": 3,
    "overall_score": 1
  },
  "metrics": {
    "retrieval_seconds": 0.11,
    "generation_seconds": 5.76,
    "total_seconds": 6.03,
    "cached": false
  }
}
```

---

## 🧪 Dataset

PubMed RCT abstracts (20k / 200k dataset).

Dataset is not included in this repository due to size constraints.

---

## 🎯 Performance Characteristics

- Semantic cache hit latency: < 100 ms
- Cold query latency: ~5–10 seconds (LLM dependent)
- Cross-encoder re-ranking improves precision over raw FAISS retrieval
- Evidence verification ensures grounded outputs

---

## 📌 Why This Project Matters

This project demonstrates:

- Production ML system design
- Vector search implementation
- Retrieval + generation integration
- Backend API engineering
- Semantic caching strategy
- Dockerized microservice deployment
- Secure API authentication
- Modular ML architecture

---

## 👩‍💻 Author

Bhavya Lakkamraju  
MS Computer Science – Data Analytics & Big Data Mining  
Focus: ML Systems, RAG Architectures, Applied AI

---

## ⭐ Future Improvements

- Hybrid BM25 + dense retrieval
- Streaming LLM responses
- Distributed FAISS index
- Cloud deployment (AWS/GCP)
- Monitoring with Prometheus/Grafana
- Automated testing suite
