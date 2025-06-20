# Curriculum AI (Git Repo-Based Textbook Processing)

This system integrates OpenBooks (which fetches textbooks as Git repositories) with an AI-based pipeline to generate comprehensive curricula and assessment questions using LangGraph, LangChain, and LLMs.

## 📦 Book Structure

Assumes you have run:
```bash
python GetOpenBooks.py
```

Which clones educational Git repositories into:
```
Curriculum-AI/Books/<language>/<discipline>/<level>/
```
Each directory is a Git repo with markdown or LaTeX textbook content.

## 💡 Key Features

- Parses and embeds full-text books from Git repos
- Extracts and organizes ~1000 subtopics per discipline using LangGraph
- Builds prerequisite-aware curriculum graphs
- Generates Bloom-level questions with answer keys
- Supports all 19 OpenAlex disciplines

## 🏁 Quickstart

```bash
python GetOpenBooks.py
python scripts/parse_textbooks.py
python scripts/embed_chunks.py
python scripts/generate_curriculum.py --discipline physics
python scripts/generate_questions.py --discipline physics
streamlit run ui/admin_app.py
```
