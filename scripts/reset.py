"""Delete call sessions (utterances, claims, flags, memos) so a demo starts clean. Dossier data stays.

    python -m scripts.reset            # calls only
    python -m scripts.reset --all      # everything in the database
"""

import sys

from app.admin import reset_all, reset_calls
from app.neo import Neo

if __name__ == "__main__":
    db = Neo()
    if "--all" in sys.argv:
        print(f"Deleted {reset_all(db)} nodes.")
    else:
        print(f"Deleted {reset_calls(db)} call nodes.")
