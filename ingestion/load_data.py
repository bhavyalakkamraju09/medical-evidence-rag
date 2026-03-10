import pandas as pd
from .parse_sections import extract_section

def load_pubmed_data(file_path):
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.startswith("#") and len(line.strip()) > 0:
                section = extract_section(line)
                rows.append({
                    "text": line.strip(),
                    "section": section
                })

    return pd.DataFrame(rows)
