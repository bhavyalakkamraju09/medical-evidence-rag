import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:!?() ]", "", text)
    return text.strip()

def preprocess_dataframe(df):
    df["clean_text"] = df["text"].apply(clean_text)
    return df
