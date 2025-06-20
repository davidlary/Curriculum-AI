import os
import json
from pathlib import Path

BOOKS_DIR = Path("Books")
CHUNKS_DIR = Path("Chunks")
CHUNKS_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = [".md", ".tex"]

def extract_chunks_from_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Simple chunking logic by headers or paragraphs
    chunks = []
    lines = text.split("\n")
    buffer = []
    for line in lines:
        if line.strip().startswith("#") and buffer:
            chunks.append("\n".join(buffer).strip())
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        chunks.append("\n".join(buffer).strip())
    return chunks

def parse_textbooks(discipline="Physics", language="english"):
    books_path = BOOKS_DIR / language / discipline
    if not books_path.exists():
        print(f"[!] Path not found: {books_path}")
        print(f"Available disciplines: {list(d.name for d in (BOOKS_DIR / language).iterdir() if d.is_dir())}")
        return

    for level_dir in books_path.iterdir():
        if level_dir.is_dir():
            level = level_dir.name
            output_file = CHUNKS_DIR / f"{discipline}_{level}.jsonl"
            with open(output_file, "w", encoding="utf-8") as out_f:
                for repo in level_dir.iterdir():
                    if repo.is_dir() and (repo / ".git").exists():
                        print(f"📘 Scanning repo: {repo}")
                        for file in repo.rglob("*"):
                            if file.suffix in SUPPORTED_EXTENSIONS:
                                chunks = extract_chunks_from_file(file)
                                for i, chunk in enumerate(chunks):
                                    record = {
                                        "discipline": discipline,
                                        "language": language,
                                        "level": level,
                                        "source_file": str(file),
                                        "chunk_index": i,
                                        "text": chunk
                                    }
                                    out_f.write(json.dumps(record) + "\n")
                        print(f"✔️ Saved chunks to {output_file}")

if __name__ == '__main__':
    parse_textbooks()