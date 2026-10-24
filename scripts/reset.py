"""Delete call sessions (utterances, claims, memos) so a demo starts clean. Dossier data stays.

    python -m scripts.reset            # sessions only
    python -m scripts.reset --all      # everything in the database
"""

import sys

from app.neo import Neo

if __name__ == "__main__":
    db = Neo()
    if "--all" in sys.argv:
        n = db.query("MATCH (n) DETACH DELETE n RETURN count(n) AS n")[0]["n"]
        print(f"Deleted {n} nodes.")
    else:
        n = db.query("""
            MATCH (s:Session) OPTIONAL MATCH (s)-[:HAS_UTTERANCE]->(u)-[:MADE_CLAIM]->(c)
            WITH collect(DISTINCT s) + collect(DISTINCT u) + collect(DISTINCT c) AS doomed
            UNWIND doomed AS d DETACH DELETE d RETURN count(d) AS n""")[0]["n"]
        print(f"Deleted {n} session nodes.")
