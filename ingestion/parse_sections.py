SECTION_LABELS = [
    "background",
    "objective",
    "methods",
    "results",
    "conclusions"
]

def extract_section(line):
    line_lower = line.lower()
    for label in SECTION_LABELS:
        if line_lower.startswith(label):
            return label
    return "unknown"
