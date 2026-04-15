"""Summarization pipeline using Gemini."""

from __future__ import annotations
import json
import os
import re
from typing import Any
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database.models import Article
from pipeline.embeddings import index_article

load_dotenv()

PROMPT_TEMPLATE = """
You are a professional news editor. Given the following news article, generate:
1. A one-liner summary under 280 characters (tweet style)
2. A concise summary in 3-5 sentences
3. Exactly 5 key bullet points

Article Title: {title}
Article Content: {content}

Respond ONLY in this exact JSON format, no extra text:
{
  "one_liner": "...",
  "short_summary": "...",
  "bullets": ["point1", "point2", "point3", "point4", "point5"]
}
"""

def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output.")
    return json.loads(text[start : end + 1])

def _fallback_summary(title: str, content: str) -> dict[str, Any]:
    """Builds a simple local summary when the LLM is unavailable."""
    clean = re.sub(r"\s+", " ", content).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    short_sentences = sentences[:5] if sentences else [clean[:500]]
    short_summary = " ".join(short_sentences).strip()[:1200]
    one_liner = f"{title}: {short_sentences[0] if short_sentences else clean[:180]}".strip()[:280]

    bullets = []
    for sentence in short_sentences[:5]:
        bullets.append(sentence[:180])
    while len(bullets) < 5:
        bullets.append("No additional key point available.")

    return {
        "one_liner": one_liner,
        "short_summary": short_summary,
        "bullets": bullets[:5],
    }

def _call_gemini(title: str, content: str) -> dict[str, Any]:
    import google.generativeai as genai
    prompt = PROMPT_TEMPLATE.format(title=title, content=content)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    output_text = (response.text or "").strip()
    return _extract_json(output_text)

def run_summarization_pipeline(db: Session) -> int:
    """Summarizes unsummarized articles and updates DB."""
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return 0

    genai.configure(api_key=api_key)
    pending_articles = db.query(Article).filter(Article.one_liner.is_(None)).all()
    updated = 0

    for article in pending_articles:
        result = None
        for attempt in range(2):  # Retry once after first failure.
            try:
                result = _call_gemini(article.title, article.content)
                break
            except Exception as exc:
                print(f"Summarization failed for article {article.id}: {exc}")
                if attempt == 1:
                    result = None

        if not result:
            result = _fallback_summary(article.title, article.content)

        try:
            bullets = result.get("bullets", [])
            if not isinstance(bullets, list):
                bullets = []
            bullets = [str(item) for item in bullets][:5]
            while len(bullets) < 5:
                bullets.append("N/A")

            article.one_liner = str(result.get("one_liner", "")).strip()
            article.short_summary = str(result.get("short_summary", "")).strip()
            article.bullets = json.dumps(bullets)
            db.add(article)
            db.commit()
            db.refresh(article)

            index_article(article)
            updated += 1
        except Exception:
            db.rollback()
            continue

    return updated
