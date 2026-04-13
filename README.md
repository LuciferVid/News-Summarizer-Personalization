1: # AI-Powered News Summarizer & Personalization Engine
2: 
3: <div align="center">
4:   <img src="https://img.shields.io/badge/Stack-FastAPI_|_Streamlit_|_FAISS-black?style=for-the-badge&logo=fastapi" />
5:   <img src="https://img.shields.io/badge/A.I.-Gemini_1.5-blue?style=for-the-badge&logo=google-gemini" />
6:   <img src="https://img.shields.io/badge/Data-RAG_|_Semantic_Search-green?style=for-the-badge" />
7: </div>
8: 
9: ---
10: 
11: This platform fetches real-time headlines, summarizes them using Google Gemini, and employs a RAG (Retrieval-Augmented Generation) workflow for high-accuracy news Q&A. It features a personalization engine that learns from user preferences to rank daily feeds.
12: 
13: ## 🌟 Key Features
14: 
15: - **Intelligent Summarization**: Generates one-liners, 3-5 sentence summaries, and 5 key bullet points for every article.
16: - **Semantic Retrieval**: Uses FAISS and `all-MiniLM-L6-v2` embeddings for meaning-based search rather than keyword matching.
17: - **RAG-Driven Q&A**: Answers user questions strictly using current news context to minimize hallucinations.
18: - **Interaction-Based Personalization**: Dynamically updates category weights based on "thumbs up/down" interactions to tailor the user feed.
19: 
20: ## 🏗️ Architecture Overview
21: 
22: 1. **Scraper**: Fetches top headlines via NewsAPI and deduplicates via SQLite.
23: 2. **Summarizer**: Parallelized Gemini calls for efficient information extraction.
24: 3. **Vector Store**: FAISS indexing for millisecond-latency semantic lookups.
25: 4. **RAG Pipeline**: Contextual retrieval and answer generation using localized news chunks.
26: 5. **Recommender**: Bayesian preference updates for personalized ranking.
27: 6. **UI**: Built with Streamlit for a rapid, interactive data experience.
28: 
29: ## 💻 Installation & Setup
30: 
31: 1. **Clone & Setup Environment**:
32:    ```bash
33:    git clone <your-repo-url>
34:    cd news_summarizer
35:    python -m venv .venv
36:    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
37:    pip install -r requirements.txt
38:    ```
39: 
40: 2. **Environment Variables**:
41:    ```bash
42:    cp .env.example .env
43:    ```
44:    Edit `.env` with your `GEMINI_API_KEY` and `NEWS_API_KEY`.
45: 
46: 3. **Run Platform**:
47:    ```bash
48:    # Terminal 1: Backend
49:    uvicorn main:app --reload
50: 
51:    # Terminal 2: Frontend
52:    streamlit run frontend/app.py
53:    ```
54: 
55: ## 🛣️ API Endpoints
56: 
57: | Method | Endpoint | Description |
58: |---|---|---|
59: | GET | `/news/feed` | Personalized feed ranking |
60: | POST | `/news/ask` | RAG-based context-aware Q&A |
61: | POST | `/user/interact` | Record preferences (Like/Skip) |
62: | GET | `/news/search` | Semantic search via FAISS |
63: | POST | `/news/refresh` | Manual sync of latest news |
64: 
65: ---
66: *Built with ❤️ for modern information consumers.*

