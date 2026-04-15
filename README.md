# AI-Powered News Summarizer & Personalization Engine

<div align="center">
  <img src="https://img.shields.io/badge/Stack-FastAPI_|_Streamlit_|_FAISS-black?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/A.I.-Gemini_2.0_Flash-blue?style=flat-square&logo=google-gemini" />
  <img src="https://img.shields.io/badge/Data-RAG_|_Semantic_Search-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Database-PostgreSQL_(Neon)-purple?style=flat-square&logo=postgresql" />
</div>

---

## Live Deployment

| Service | URL |
|---|---|
| **Backend API** | [news-summarizer-personalization.onrender.com](https://news-summarizer-personalization.onrender.com) |
| **Health Check** | [/health](https://news-summarizer-personalization.onrender.com/health) |

---

This enterprise-grade news intelligence platform orchestrates real-time headline ingestion, LLM-based summarization (Google Gemini), and high-fidelity Retrieval-Augmented Generation (RAG). The system features a sophisticated recommendation engine that optimizes user feeds based on multi-dimensional interaction signals.

## Core Capabilities

- **Intelligent Analytical Summaries**: Automated generation of concise one-liners, executive summaries, and structured bullet points for high-density information processing.
- **Semantic Vector Retrieval**: Utilizes FAISS indexing and `all-MiniLM-L6-v2` transformer embeddings for high-dimensional semantic search indexing.
- **Context-Aware RAG Pipeline**: Implements a strict retrieval-augmented generation workflow to ensure news-based Q&A remains grounded in factual source material.
- **Preference-Driven Personalization**: Employs a Bayesian weighting system to dynamically adjust category relevance based on implicit and explicit user interaction behavior.
- **Lazy-Loaded ML Infrastructure**: Production-optimized deferred model initialization to ensure sub-second cold starts on cloud platforms.

## System Architecture

1. **Data Ingestion**: Distributed scraper architecture for real-time news retrieval with relational deduplication via GNews API.
2. **Analysis Pipeline**: Asynchronous LLM processing (Gemini 2.0 Flash) for feature extraction and summarization.
3. **Vector Infrastructure**: FAISS-backed persistence layer for low-latency semantic indexing.
4. **Persistence Layer**: PostgreSQL (Neon) for production-grade relational storage with SQLite fallback for local development.
5. **Logic Tier**: FastAPI-based microservice architecture for RAG orchestration and personalized ranking.
6. **Presentation Layer**: Responsive Streamlit dashboard for real-time data visualization and interaction.

## Setup and Configuration

### Environment Initialization
```bash
git clone https://github.com/LuciferVid/News-Summarizer-Personalization.git
cd news_summarizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
```
Ensure you provide valid API keys for `GEMINI_API_KEY` and `GNEWS_API_KEY` within the `.env` file.

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
| GET | `/health` | Returns service health and environment configuration status |
| GET | `/news/feed` | Retrieves ranked personalized news items |
| POST | `/news/ask` | Executes RAG-based context-aware queries |
| POST | `/user/interact` | Ingests interaction telemetry for personalization |
| GET | `/news/search` | Performs high-dimensional semantic search |
| POST | `/news/refresh` | Manual execution of the synchronization pipeline |

## Deployment

- **Backend**: Render (Web Service)
- **Database**: Neon PostgreSQL
- **Frontend**: Streamlit Cloud

### Required Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (Neon) |
| `GEMINI_API_KEY` | Google Gemini API key for summarization and RAG |
| `GNEWS_API_KEY` | GNews API key for real-time news ingestion |

---
*News Summarizer & Personalization Engine - 2026*
