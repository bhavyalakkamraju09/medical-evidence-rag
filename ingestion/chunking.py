def chunk_dataframe(df, chunk_size, overlap):
    all_chunks = []

    for idx, row in df.iterrows():
        words = row["clean_text"].split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])

            all_chunks.append({
                "chunk": chunk,
                "source_id": idx,
                "section": row["section"]
            })

    return all_chunks
