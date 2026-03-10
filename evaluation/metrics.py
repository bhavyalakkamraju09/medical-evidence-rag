def precision_at_k(results, keywords):

    if not results:
        return 0.0

    relevant_count = 0

    for r in results:
        for kw in keywords:
            if kw.lower() in r["text"]:
                relevant_count += 1
                break

    return relevant_count / len(results)
