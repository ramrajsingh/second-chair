"""Neo4j connection and query helper. Queries can be logged per session so the UI can cite
the exact Cypher behind every verdict."""

import time
from typing import Any

from neo4j import GraphDatabase

from app import config


class GraphError(RuntimeError):
    pass


class Neo:
    def __init__(self):
        if not config.NEO4J_URI:
            raise GraphError("NEO4J_URI is not set in .env")
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))

    def query(self, cypher: str, params: dict | None = None, log: list | None = None,
              label: str = "") -> list[dict[str, Any]]:
        t = time.time()
        try:
            records, _, _ = self.driver.execute_query(cypher, params or {}, database_=config.NEO4J_DATABASE)
        except Exception as e:
            raise GraphError(f"{label or 'query'} failed: {e}") from e
        if log is not None:
            log.append({"label": label, "cypher": " ".join(cypher.split()), "ms": int((time.time() - t) * 1000)})
        return [r.data() for r in records]


SCHEMA = [
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.pid IS UNIQUE",
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.key IS UNIQUE",
    "CREATE CONSTRAINT investor_name IF NOT EXISTS FOR (i:Investor) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE INDEX headcount_date IF NOT EXISTS FOR (h:HeadcountPoint) ON (h.date)",
]


def setup_schema(db: Neo):
    for stmt in SCHEMA:
        db.query(stmt)


def company_key(name: str | None, domain: str | None = None) -> str:
    """Identity for companies: the domain when known, else the normalized name."""
    if domain:
        return domain.lower().removeprefix("www.").strip("/")
    return " ".join((name or "").lower().split())
