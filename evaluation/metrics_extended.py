def recall_at_k(results, relevant_keywords):
    relevant = 0
    for r in results:
        if any(word in r["text"].lower() for word in relevant_keywords):
            relevant += 1
    return relevant / len(relevant_keywords)


def hit_rate(results, relevant_keywords):
    for r in results:
        if any(word in r["text"].lower() for word in relevant_keywords):
            return 1
    return 0


def mrr(results, relevant_keywords):
    for i, r in enumerate(results):
        if any(word in r["text"].lower() for word in relevant_keywords):
            return 1 / (i + 1)
    return 0