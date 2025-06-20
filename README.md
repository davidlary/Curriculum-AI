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
# Set your API keys (at minimum OpenAI for embeddings)
export OPENAI_API_KEY='your-openai-key-here'
export ANTHROPIC_API_KEY='your-anthropic-key-here'  # Optional
export XAI_API_KEY='your-xai-key-here'              # Optional

# Run the pipeline
python GetOpenBooks.py
python scripts/parse_textbooks.py
python scripts/embed_chunks.py --provider openai
python scripts/generate_curriculum.py --discipline physics --provider anthropic
python scripts/generate_questions.py --discipline physics --provider xai
streamlit run ui/admin_app.py
```

## 🤖 AI Provider Support

- **OpenAI**: GPT-4, GPT-3.5-turbo + text-embedding-ada-002
- **Anthropic**: Claude-3-Sonnet, Claude-3-Haiku  
- **XAI**: Grok-beta (OpenAI-compatible API)

*Note: OpenAI API key required for embeddings regardless of other providers chosen.*

