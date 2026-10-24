"""Resetting the database between demos (used by the Reset buttons and scripts/reset.py)."""

from app.neo import Neo

CALL_LABELS = ["Session", "Utterance", "Claim", "Flag"]


def reset_calls(db: Neo) -> int:
    """Delete every call (utterances, claims, evasion flags, memos). Dossier data stays."""
    return db.query("""
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)
        DETACH DELETE n RETURN count(n) AS n""", {"labels": CALL_LABELS})[0]["n"]


def reset_all(db: Neo) -> int:
    """Empty the database. Cached Crustdata responses on disk are kept, so rebuilding costs no credits."""
    return db.query("MATCH (n) DETACH DELETE n RETURN count(n) AS n")[0]["n"]
