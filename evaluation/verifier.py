import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Load lightweight embedding model for verification
verifier_model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_answer(answer, evidence_chunks, threshold=0.6):

    # Split into sentences
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if len(sentences) == 0:
        return {
            "verified_sentences": 0,
            "flagged_sentences": 0,
            "overall_score": 0.0
        }

    # Embed sentences
    sentence_embeddings = verifier_model.encode(sentences)
    faiss.normalize_L2(sentence_embeddings)

    # Embed evidence
    evidence_texts = [chunk["text"] for chunk in evidence_chunks]
    evidence_embeddings = verifier_model.encode(evidence_texts)
    faiss.normalize_L2(evidence_embeddings)

    # Build temporary index
    dimension = evidence_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(evidence_embeddings)

    verified = 0
    flagged = 0

    for emb in sentence_embeddings:
        emb = np.expand_dims(emb, axis=0)
        scores, _ = index.search(emb, 1)

        similarity = scores[0][0]

        if similarity >= threshold:
            verified += 1
        else:
            flagged += 1

    overall_score = verified / len(sentences)

    return {
        "verified_sentences": verified,
        "flagged_sentences": flagged,
        "overall_score": round(overall_score, 2)
    }