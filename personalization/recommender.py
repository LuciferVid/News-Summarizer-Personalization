"""User preference updates and personalized feed ranking."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from database.models import Article, UserPreference

CATEGORIES = ["technology", "sports", "business", "health", "entertainment"]
DEFAULT_WEIGHT = 0.2


def _ensure_preferences(db: Session, user_id: str) -> list[UserPreference]:
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).all()
    if prefs:
        return prefs

    created = []
    for category in CATEGORIES:
        pref = UserPreference(
            user_id=user_id,
            category=category,
            weight=DEFAULT_WEIGHT,
            last_updated=datetime.utcnow(),
        )
        db.add(pref)
        created.append(pref)
    db.commit()
    return created


def _normalize_preferences(prefs: list[UserPreference]) -> None:
    total = sum(max(pref.weight, 0.0) for pref in prefs)
    if total <= 0:
        for pref in prefs:
            pref.weight = DEFAULT_WEIGHT
        return
    for pref in prefs:
        pref.weight = max(pref.weight, 0.0) / total


def update_preference(db: Session, user_id: str, category: str, interaction_type: str) -> None:
    """Adjusts category preference based on read/liked/skipped interactions."""
    prefs = _ensure_preferences(db, user_id)
    target = next((pref for pref in prefs if pref.category == category), None)
    if not target:
        return

    if interaction_type in {"read", "liked"}:
        target.weight += 0.05
    elif interaction_type == "skipped":
        target.weight -= 0.02

    target.last_updated = datetime.utcnow()
    _normalize_preferences(prefs)
    db.commit()


def get_personalized_feed(db: Session, user_id: str) -> list[Article]:
    """Returns top-ranked personalized feed from latest 50 articles."""
    prefs = _ensure_preferences(db, user_id)
    weight_map = {pref.category: pref.weight for pref in prefs}

    latest_articles = (
        db.query(Article)
        .order_by(Article.published_at.desc())
        .limit(50)
        .all()
    )

    now = datetime.utcnow()
    scored: list[tuple[float, Article]] = []
    for article in latest_articles:
        hours_since = max((now - article.published_at).total_seconds() / 3600.0, 0.0)
        recency_score = 1.0 / (hours_since + 1.0)
        category_weight = weight_map.get(article.category, 0.0)
        final_score = category_weight * recency_score
        scored.append((final_score, article))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [article for _, article in scored[:20]]

