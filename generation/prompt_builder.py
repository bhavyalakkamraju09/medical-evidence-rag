def build_rag_prompt(query, retrieved_chunks):

    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks):
        truncated_text = chunk["text"][:400]
        context_blocks.append(f"[{i+1}] {truncated_text}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are a medical research assistant.

STRICT RULES:
- You MUST only copy or directly rephrase content from the evidence.
- You MUST NOT introduce medical terms that are not present in the evidence.
- If a treatment name does not appear in evidence, DO NOT mention it.
- Cite evidence only using [1], [2], etc.
- If unsure, say "Insufficient evidence."

Question:
{query}

Evidence:
{context_text}

Answer:
"""

    return prompt