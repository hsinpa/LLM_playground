def read_full_text(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()