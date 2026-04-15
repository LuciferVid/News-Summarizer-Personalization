# AI-Powered News Summarizer & Personalization Engine

<div align="center">
  <img src="https://img.shields.io/badge/Stack-FastAPI_|_Streamlit_|_FAISS-black?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/A.I.-Gemini_1.5-blue?style=flat-square&logo=google-gemini" />
  <img src="https://img.shields.io/badge/Data-RAG_|_Semantic_Search-green?style=flat-square" />
</div>

---

This enterprise-grade news intelligence platform orchestrates real-time headline ingestion, LLM-based summarization (Google Gemini), and high-fidelity Retrieval-Augmented Generation (RAG). The system features a sophisticated recommendation engine that optimizes user feeds based on multi-dimensional interaction signals.

## Core Capabilities

- **Intelligent Analytical Summaries**: Automated generation of concise one-liners, executive summaries, and structured bullet points for high-density information processing.
- **Semantic Vector Retrieval**: Utilizes FAISS indexing and `all-MiniLM-L6-v2` transformer embeddings for high-dimensional semantic search indexing.
- **Context-Aware RAG Pipeline**: Implements a strict retrieval-augmented generation workflow to ensure news-based Q&A remains grounded in factual source material.
- **Preference-Driven Personalization**: Employs a Bayesian weighting system to dynamically adjust category relevance based on implicit and explicit user interaction behavior.

## System Architecture

1. **Data Ingestion**: Distributed scraper architecture for real-time news retrieval with relational deduplication.
2. **Analysis Pipeline**: Asynchronous LLM processing for feature extraction and summarization.
3. **Vector Infrastructure**: FAISS-backed persistence layer for low-latency semantic indexing.
4. **Logic Tier**: FastAPI-based microservice architecture for RAG orchestration and personalized ranking.
5. **Presentation Layer**: Responsive Streamlit dashboard for real-time data visualization and interaction.

## Setup and Configuration

### Environment Initialization
```bash
git clone <repository-url>
cd news_summarizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
```
Ensure you provide valid API keys for `GEMINI_API_KEY` and `NEWS_API_KEY` within the `.env` file.

### Execution
```bash
# Instance 1: Backend API
uvicorn main:app --reload

# Instance 2: Frontend Dashboard
streamlit run frontend/app.py
```

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| GET | `/news/feed` | Retrieves ranked personalized news items |
| POST | `/news/ask` | Executes RAG-based context-aware queries |
| POST | `/user/interact` | Ingests interaction telemetry for personalization |
| GET | `/news/search` | Performs high-dimensional semantic search |
| POST | `/news/refresh` | Manual execution of the synchronization pipeline |

---
*News Summarizers & Personalization Documentation - 2026*
