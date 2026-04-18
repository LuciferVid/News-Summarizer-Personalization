"""Streamlit frontend for personalized news, RAG Q&A, and search."""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL", "https://news-summarizer-personalization.onrender.com"
)
CATEGORY_COLORS = {
    "technology": "#1f77b4",
    "sports": "#2ca02c",
    "business": "#9467bd",
    "health": "#d62728",
    "entertainment": "#ff7f0e",
}


def _safe_get(url: str, params: dict | None = None) -> list | dict:
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return [] if "feed" in url or "search" in url else {}


def _safe_post(url: str, payload: dict) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return {}


def render_news_feed(user_id: str) -> None:
    st.title("News Feed")
    articles = _safe_get(f"{BACKEND_URL}/news/feed", params={"user_id": user_id})
    if not isinstance(articles, list):
        st.warning("Unable to load feed.")
        return

    all_categories = ["technology", "sports", "business", "health", "entertainment", "world", "nation"]
    selected_categories = []
    st.sidebar.subheader("Category Filters")
    for category in all_categories:
        if st.sidebar.checkbox(category.title(), value=True, key=f"filter_{category}"):
            selected_categories.append(category)

    filtered = [article for article in articles if article.get("category") in selected_categories]

    if not filtered:
        st.info("No news found in the current selection.")
        st.markdown("""
        ### Why is this empty?
        1. **Cold Start:** The system just woke up and is still fetching news in the background. Please wait ~60 seconds and refresh.
        2. **API Quota:** The daily news API limit may have been reached.
        3. **Filters:** You might have unchecked all categories in the sidebar.
        
        Click the **🔄 Refresh News** button in the sidebar to manually trigger an update.
        """)
        return

    col1, col2 = st.columns(2)
    for idx, article in enumerate(filtered):
        column = col1 if idx % 2 == 0 else col2
        with column:
            st.markdown(f"**{article.get('title', 'Untitled')}**")
            st.caption(f"{article.get('source', 'Unknown')} | {article.get('published_at', 'N/A')}")
            st.write(article.get("one_liner") or "Summary pending...")
            category = article.get("category", "unknown")
            color = CATEGORY_COLORS.get(category, "#777777")
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 8px;border-radius:8px;'>{category}</span>",
                unsafe_allow_html=True,
            )

            with st.expander("Read More"):
                st.write(article.get("short_summary") or "Detailed summary pending...")
                bullets = article.get("bullets", [])
                for bullet in bullets:
                    st.markdown(f"- {bullet}")

            up_col, down_col = st.columns(2)
            if up_col.button("👍", key=f"like_{article.get('id')}"):
                _safe_post(
                    f"{BACKEND_URL}/user/interact",
                    {
                        "user_id": user_id,
                        "article_id": article.get("id"),
                        "interaction_type": "liked",
                    },
                )
                st.success("Preference updated.")

            if down_col.button("👎", key=f"skip_{article.get('id')}"):
                _safe_post(
                    f"{BACKEND_URL}/user/interact",
                    {
                        "user_id": user_id,
                        "article_id": article.get("id"),
                        "interaction_type": "skipped",
                    },
                )
                st.success("Preference updated.")

            st.markdown(f"[Open Article]({article.get('url', '#')})")
            st.divider()


def render_ask_news(user_id: str) -> None:
    st.title("Ask News (RAG)")
    query = st.text_input("Ask anything about today's news...")
    if st.button("Submit Question"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        result = _safe_post(f"{BACKEND_URL}/news/ask", {"query": query, "user_id": user_id})
        if not result:
            return

        st.info(result.get("answer", "No answer found."))
        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources")
            for source in sources:
                st.markdown(f"- [{source.get('title', 'Untitled')}]({source.get('url', '#')})")


def render_search() -> None:
    st.title("Semantic Search")
    keyword = st.text_input("Search by meaning (not exact keywords only)")
    if st.button("Search"):
        if not keyword.strip():
            st.warning("Please enter a search query.")
            return

        results = _safe_get(f"{BACKEND_URL}/news/search", params={"q": keyword})
        if not isinstance(results, list) or not results:
            st.warning("No matching results found.")
            return

        for article in results:
            st.markdown(f"**{article.get('title', 'Untitled')}**")
            st.caption(f"{article.get('source', 'Unknown')} | {article.get('published_at', datetime.utcnow().isoformat())}")
            st.write(article.get("one_liner") or "Summary pending...")
            st.markdown(f"[Open Article]({article.get('url', '#')})")
            st.divider()


def main() -> None:
    st.set_page_config(page_title="AI News Summarizer", layout="wide")
    st.sidebar.title("Navigation")
    user_id = st.sidebar.text_input("User ID", value="user_1")
    page = st.sidebar.radio("Go to", ["News Feed", "Ask News", "Search"])

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh News"):
        with st.spinner("Fetching and summarizing latest news..."):
            refresh_result = _safe_post(f"{BACKEND_URL}/news/refresh", {})
            if refresh_result:
                st.sidebar.success(f"Added {refresh_result.get('new_articles', 0)} articles.")
                st.rerun()

    if page == "News Feed":
        render_news_feed(user_id)
    elif page == "Ask News":
        render_ask_news(user_id)
    else:
        render_search()


if __name__ == "__main__":
    main()

