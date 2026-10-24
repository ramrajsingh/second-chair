"""The live loop: spot claims in the subject's speech, verify each against the dossier graph,
and whisper a follow-up only on a high-confidence contradiction.

    utterance --LLM (fast, structured)--> claims --Cypher vs dossier--> verdict + evidence
              --(contradicted)--> whisper text --> TTS in the interviewer's ear

Every claim, its verdict, evidence and the Cypher used are written back to the graph:
    (:Session)-[:HAS_UTTERANCE]->(:Utterance)-[:MADE_CLAIM]->(:Claim)
"""

import json
import re
import uuid
from datetime import date

from openai import OpenAI

from app import config
from app.neo import Neo

CLAIM_TYPES = ["headcount", "growth", "role", "funding_total", "funding_round", "investor", "hiring", "shared_history"]

EXTRACT_PROMPT = """You listen to a founder or candidate on a diligence call and extract factual claims that can be
checked against company and people data. Only extract concrete, checkable claims of these types:
- headcount: current team size ("we're 200 people") -> number
- growth: team growth over a period ("we tripled the team this year") -> number = multiplier (3 for tripled, 1.5 for +50%), period_months
- role: a past or current job ("I was VP Engineering at Flipkart from 2018 to 2021") -> company, title, start_year, end_year
- funding_total: total raised ("we've raised $20M") -> number in USD
- funding_round: a specific round ("our Series A was $12M") -> round_type, number in USD
- investor: a named backer ("Peak Harbor led our A") -> investor
- hiring: hiring activity ("we're hiring 30 engineers") -> number of open roles
- shared_history: worked with someone before ("my co-founder and I worked together at Flipkart") -> person, company
Company-level claims (headcount, growth, funding, investor, hiring) can be about another company the speaker
names, such as a competitor ("Shiprocket only has 50 people"); put that company in `company`, otherwise leave
`company` null for claims about the speaker's own company.
Also list in `mentions` every other company or person the speaker names (competitors, customers, past employers,
advisors, investors, colleagues), except the speaker and their own company.
Extract claims ONLY from the Utterance. The context is only for resolving references like "we", "she" or
"the company"; never re-extract claims that appear in the context.
Ignore opinions, plans and vague statements. Return an empty list if there are no checkable claims.
Use null for unknown fields. The speaker may mix Hindi and English; output English."""

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": CLAIM_TYPES},
            "statement": {"type": "string", "description": "The claim in a short English sentence"},
            "number": {"type": ["number", "null"]},
            "period_months": {"type": ["integer", "null"]},
            "company": {"type": ["string", "null"]},
            "title": {"type": ["string", "null"]},
            "start_year": {"type": ["integer", "null"]},
            "end_year": {"type": ["integer", "null"]},
            "round_type": {"type": ["string", "null"]},
            "investor": {"type": ["string", "null"]},
            "person": {"type": ["string", "null"]},
        },
        "required": ["type", "statement", "number", "period_months", "company", "title", "start_year",
                     "end_year", "round_type", "investor", "person"],
        "additionalProperties": False,
    }}},
    "required": ["claims", "mentions"],
    "additionalProperties": False,
}
CLAIM_SCHEMA["properties"]["mentions"] = {"type": "array", "items": {
    "type": "object",
    "properties": {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["company", "person"]}},
    "required": ["name", "kind"], "additionalProperties": False}}

WHISPER_PROMPT = """You whisper into an interviewer's ear during a live diligence call. Given a claim the subject just
made and the contradicting data, write ONE sentence of at most 25 words: what the data shows, then a sharp follow-up
question to ask. No preamble. Example: "He says 200 people; data shows 84, down 12% this year. Ask what drove attrition." """

DEMO_LINES_PROMPT = """You write test lines for demoing a diligence copilot. Given a dossier of real company and founder
data, write 6 short things the founder might say on an investor call, in the first person, natural and spoken:
- 3 lines that are ACCURATE per the dossier (e.g. funding total, a real investor, a real past role, co-founder overlap)
- 3 lines that are INFLATED or wrong (e.g. headcount about double the real number, "we tripled the team" when growth
  is flat or negative, a more senior title at a real past employer, a funding amount well above the real total)
Use only facts present in the dossier; skip claim types the dossier has no data for. Mix the order. Make one line
Hinglish (Hindi in Latin script mixed with English). Return JSON: {"lines": ["...", ...]}."""

MEMO_PROMPT = """Write a concise diligence memo in Markdown from the call record below. Sections: Summary (2-3 sentences),
Claims checked (a table: claim, verdict, evidence with source), Red flags, Questions for the next call. Every factual
statement must cite its source as [Crustdata: <field>] or [Call: <speaker quote>]. No speculation beyond the data."""

COMPANY_LEVEL = {"headcount", "growth", "funding_total", "funding_round", "investor", "hiring"}

SENIORITY = [("intern", 0), ("engineer", 1), ("developer", 1), ("analyst", 1), ("senior", 2), ("lead", 3),
             ("principal", 3), ("staff", 3), ("manager", 3), ("head", 4), ("director", 5), ("vp", 6),
             ("vice president", 6), ("chief", 7), ("cto", 7), ("ceo", 7), ("coo", 7), ("founder", 7)]


def _seniority(title: str) -> int:
    t = (title or "").lower()
    return max([lvl for word, lvl in SENIORITY if re.search(rf"\b{word}\b", t)], default=1)


def signature(c: dict) -> tuple:
    """Identity of a claim, so the same claim repeated later in a call is checked (and whispered) once."""
    norm = lambda v: " ".join(str(v).lower().split()) if v is not None else None
    t = "investor" if c["type"] == "funding_round" and c["number"] is None and c["investor"] else c["type"]
    if t in ("headcount", "growth", "funding_total", "hiring"):
        # Same type with a different number, or about another company, is a new claim.
        return (t, norm(c["company"]),
                round(c["number"], -1 if t == "headcount" else 0) if c["number"] is not None else None)
    return (t, norm(c["company"]), norm(c["investor"]), norm(c["round_type"]), norm(c["person"]))


def _usd(n: float | None) -> str:
    if n is None:
        return "unknown"
    return f"${n / 1e6:.1f}M" if n >= 1e6 else f"${n:,.0f}"


class ClaimEngine:
    def __init__(self, db: Neo):
        self.db = db
        self.llm = OpenAI(api_key=config.OPENAI_API_KEY or None)

    # --- Extraction ------------------------------------------------------
    def extract(self, text: str, context: str) -> dict:
        """Returns {"claims": [...], "mentions": [{"name", "kind"}]}."""
        resp = self.llm.chat.completions.create(
            model=config.OPENAI_FAST_MODEL,
            messages=[{"role": "system", "content": EXTRACT_PROMPT},
                      {"role": "user", "content": f"Context: {context}\nUtterance: {text}"}],
            response_format={"type": "json_schema", "json_schema": {"name": "claims", "strict": True,
                                                                    "schema": CLAIM_SCHEMA}})
        out = json.loads(resp.choices[0].message.content or "{}")
        return {"claims": out.get("claims", []), "mentions": out.get("mentions", [])}

    # --- Verification: each claim type is one Cypher read on the dossier ----
    def verify(self, c: dict, ckey: str, pid: str | None, log: list, scope: dict | None = None) -> dict:
        """scope: companies {key: name} and people [pid] in this call's dossier, including ones added mid-call."""
        scope = scope or {"companies": {}, "people": []}
        if c["type"] == "funding_round" and c["number"] is None and c["investor"]:
            c = {**c, "type": "investor"}  # "X led our Series A" names a backer, not an amount
        target = ckey
        if c["type"] in COMPANY_LEVEL and c["company"]:
            named = self._resolve_company(c["company"], scope["companies"])
            if named is None:
                return {"status": "unverifiable", "target": None, "evidence": [],
                        "finding": f"{c['company']} isn't in the dossier yet; add it to check this claim."}
            target = named
        fn = getattr(self, f"_check_{c['type']}", None)
        if not fn:
            return {"status": "unverifiable", "finding": "No data source for this claim type.", "evidence": [],
                    "target": target}
        return {**fn({**c, "_people": scope["people"]}, target, pid, log), "target": target}

    @staticmethod
    def _resolve_company(name: str, companies: dict) -> str | None:
        n = re.sub(r"\s*\(.*?\)", "", name).strip().lower()
        for key, cname in companies.items():
            c = re.sub(r"\s*\(.*?\)", "", cname or "").strip().lower()
            if n and c and (n in c or c in n or n == key):
                return key
        return None

    def _check_headcount(self, c, ckey, pid, log):
        rows = self.db.query("""
            MATCH (co:Company {key: $ckey})-[:HEADCOUNT]->(h:HeadcountPoint)
            RETURN co.name AS company, h.count AS count, toString(h.date) AS date ORDER BY h.date DESC LIMIT 1""",
                             {"ckey": ckey}, log, "headcount: latest point")
        if not rows or c["number"] is None:
            return {"status": "unverifiable", "finding": "No headcount data.", "evidence": []}
        actual, claimed = rows[0]["count"], c["number"]
        ok = abs(claimed - actual) / max(actual, 1) <= 0.2
        return {"status": "verified" if ok else "contradicted",
                "finding": f"Claimed {claimed:g}; Crustdata headcount is {actual} as of {rows[0]['date']}.",
                "evidence": [{"source": "Crustdata headcount.timeseries", "value": f"{actual} on {rows[0]['date']}"}]}

    def _check_growth(self, c, ckey, pid, log):
        months = c["period_months"] or 12
        rows = self.db.query("""
            MATCH (co:Company {key: $ckey})-[:HEADCOUNT]->(h:HeadcountPoint)
            WITH h ORDER BY h.date DESC
            WITH collect(h) AS pts
            WITH pts[0] AS now, [p IN pts WHERE p.date <= pts[0].date - duration({months: $months})] AS older
            RETURN now.count AS now_count, toString(now.date) AS now_date,
                   older[0].count AS then_count, toString(older[0].date) AS then_date""",
                             {"ckey": ckey, "months": months}, log, f"growth over {months} months")
        if not rows or rows[0]["then_count"] is None or c["number"] is None:
            return {"status": "unverifiable", "finding": "Not enough headcount history.", "evidence": []}
        r = rows[0]
        actual = r["now_count"] / max(r["then_count"], 1)
        ok = actual >= c["number"] * 0.75
        return {"status": "verified" if ok else "contradicted",
                "finding": f"Claimed {c['number']:g}x growth; headcount went {r['then_count']} ({r['then_date']}) → "
                           f"{r['now_count']} ({r['now_date']}), {actual:.2f}x.",
                "evidence": [{"source": "Crustdata headcount.timeseries",
                              "value": f"{r['then_count']} → {r['now_count']}"}]}

    def _check_role(self, c, ckey, pid, log):
        if not pid or not c["company"]:
            return {"status": "unverifiable", "finding": "No subject profile or company named.", "evidence": []}
        rows = self.db.query("""
            MATCH (p:Person {pid: $pid})-[w:WORKED_AT]->(co:Company)
            WHERE toLower(co.name) CONTAINS toLower($company) OR toLower($company) CONTAINS toLower(co.name)
            RETURN co.name AS company, w.title AS title, w.start AS start, w.end AS end""",
                             {"pid": pid, "company": c["company"]}, log, "role: employment history")
        if not rows:
            return {"status": "contradicted", "finding": f"No employment at {c['company']} in the profile.",
                    "evidence": [{"source": "Crustdata experience.employment_details", "value": "no match"}]}
        r = rows[0]
        problems = []
        if c["title"] and _seniority(c["title"]) > _seniority(r["title"]):
            problems.append(f"title was {r['title']}, not {c['title']}")
        start_y = int(r["start"][:4]) if r["start"] else None
        end_y = int(r["end"][:4]) if r["end"] else date.today().year
        if c["start_year"] and start_y and abs(c["start_year"] - start_y) > 1:
            problems.append(f"started {start_y}, not {c['start_year']}")
        if c["end_year"] and end_y and abs(c["end_year"] - end_y) > 1:
            problems.append(f"left {end_y}, not {c['end_year']}")
        dates = f"{r['start'] or '?'} to {r['end'] or 'present'}"
        return {"status": "contradicted" if problems else "verified",
                "finding": (f"At {r['company']}: " + "; ".join(problems) + f" ({dates}).") if problems
                           else f"{r['title']} at {r['company']}, {dates}.",
                "evidence": [{"source": "Crustdata experience.employment_details",
                              "value": f"{r['title']} at {r['company']}, {dates}"}]}

    def _check_funding_total(self, c, ckey, pid, log):
        rows = self.db.query("MATCH (co:Company {key: $ckey}) RETURN co.total_funding_usd AS total",
                             {"ckey": ckey}, log, "funding: total raised")
        total = rows[0]["total"] if rows else None
        if total is None or c["number"] is None:
            return {"status": "unverifiable", "finding": "No funding data.", "evidence": []}
        ok = abs(c["number"] - total) / max(total, 1) <= 0.2
        return {"status": "verified" if ok else "contradicted",
                "finding": f"Claimed {_usd(c['number'])} raised; Crustdata shows {_usd(total)} total.",
                "evidence": [{"source": "Crustdata funding.total_investment_usd", "value": _usd(total)}]}

    def _check_funding_round(self, c, ckey, pid, log):
        rows = self.db.query("""
            MATCH (:Company {key: $ckey})-[:RAISED]->(f:FundingRound)
            WHERE $round IS NULL OR toLower(replace(f.round_type, '_', ' ')) CONTAINS toLower(replace($round, '_', ' '))
            RETURN f.round_type AS round, f.amount_usd AS amount, toString(f.date) AS date ORDER BY f.date DESC""",
                             {"ckey": ckey, "round": c["round_type"]}, log, "funding: rounds of this type")
        label = (c["round_type"] or "that").replace("_", " ")
        if not rows:  # a named round that isn't in the data at all
            all_rounds = self.db.query("""MATCH (:Company {key: $ckey})-[:RAISED]->(f:FundingRound)
                RETURN collect(f.round_type + ' ' + toString(f.date)) AS rounds""", {"ckey": ckey})[0]["rounds"]
            return {"status": "contradicted" if c["round_type"] and all_rounds else "unverifiable",
                    "finding": f"No {label} round in the data" + (f"; rounds on record: {', '.join(all_rounds)}." if all_rounds else "."),
                    "evidence": [{"source": "Crustdata funding.milestones", "value": ", ".join(all_rounds) or "none"}]}
        listed = "; ".join(f"{r['round']} {_usd(r['amount'])} on {r['date']}" for r in rows)
        if c["number"] is None:
            return {"status": "verified", "finding": f"{label.title()} round on record: {listed}.",
                    "evidence": [{"source": "Crustdata funding.milestones", "value": listed}]}
        # Compare with the latest round of that type and with the sum of all of them ("$12M over two seeds").
        total = sum(r["amount"] or 0 for r in rows)
        close = lambda a: a and abs(c["number"] - a) / a <= 0.2
        ok = close(rows[0]["amount"]) or close(total)
        return {"status": "verified" if ok else "contradicted",
                "finding": f"Claimed {_usd(c['number'])} in {label}; on record: {listed}"
                           + (f" (total {_usd(total)})." if len(rows) > 1 else "."),
                "evidence": [{"source": "Crustdata funding.milestones", "value": listed}]}

    def _check_investor(self, c, ckey, pid, log):
        rows = self.db.query("""
            MATCH (:Company {key: $ckey})-[:RAISED]->(f:FundingRound)<-[r:INVESTED_IN]-(i:Investor)
            RETURN i.name AS investor, r.lead AS lead, f.round_type AS round""", {"ckey": ckey}, log, "funding: investors")
        if not rows or not c["investor"]:
            return {"status": "unverifiable", "finding": "No investor data.", "evidence": []}
        hit = [r for r in rows if c["investor"].lower() in r["investor"].lower() or r["investor"].lower() in c["investor"].lower()]
        names = sorted({r["investor"] for r in rows})
        if hit:
            h = hit[0]
            return {"status": "verified", "finding": f"{h['investor']} invested in the {h['round']}{' as lead' if h['lead'] else ''}.",
                    "evidence": [{"source": "Crustdata funding.milestones.investors", "value": h["investor"]}]}
        return {"status": "contradicted", "finding": f"{c['investor']} is not among the investors: {', '.join(names)}.",
                "evidence": [{"source": "Crustdata funding.investors", "value": ", ".join(names)}]}

    def _check_hiring(self, c, ckey, pid, log):
        rows = self.db.query("MATCH (co:Company {key: $ckey}) RETURN co.openings AS openings, co.growth_6m_pct AS six",
                             {"ckey": ckey}, log, "hiring: open roles")
        if not rows or rows[0]["openings"] is None:
            return {"status": "unverifiable", "finding": "No hiring data.", "evidence": []}
        o, six = rows[0]["openings"], rows[0]["six"]
        ok = c["number"] is None or o >= c["number"] * 0.5
        return {"status": "verified" if ok else "contradicted",
                "finding": f"Claimed {c['number'] or 'aggressive'} hiring; {o} open roles listed"
                           + (f", headcount {six:+.0f}% over 6 months." if six is not None else "."),
                "evidence": [{"source": "Crustdata hiring", "value": f"{o} openings"}]}

    def _check_shared_history(self, c, ckey, pid, log):
        # Multi-hop: subject and another founder at the same company with overlapping dates.
        rows = self.db.query("""
            MATCH (p:Person {pid: $pid})-[a:WORKED_AT]->(co:Company)<-[b:WORKED_AT]-(o:Person)
            WHERE o <> p AND co.key <> $ckey  // before this company, not at it
              AND (o.pid IN $people OR (o)-[:FOUNDED]->(:Company {key: $ckey}))  // co-founders or people added to the dossier
              AND ($company IS NULL OR toLower(co.name) CONTAINS toLower($company))
              AND ($person IS NULL OR toLower(o.name) CONTAINS toLower($person) OR $person =~ '(?i).*(co-?founder|cto|ceo).*')
              AND a.start <= coalesce(b.end, '9999') AND b.start <= coalesce(a.end, '9999')
            RETURN o.name AS other, co.name AS company, a.start AS a_start, a.end AS a_end, b.start AS b_start, b.end AS b_end""",
                             {"pid": pid, "ckey": ckey, "company": c["company"], "person": c["person"],
                              "people": c.get("_people", [])},
                             log, "shared history: overlapping tenure (multi-hop)")
        if rows:
            r = rows[0]
            return {"status": "verified",
                    "finding": f"Overlapped with {r['other']} at {r['company']} ({max(r['a_start'], r['b_start'])[:7]} to "
                               f"{min(r['a_end'] or '9999', r['b_end'] or '9999')[:7]}).",
                    "evidence": [{"source": "Crustdata experience (both founders)", "value": f"{r['company']} overlap"}]}
        return {"status": "contradicted", "finding": "No overlapping tenure found between the founders.",
                "evidence": [{"source": "Crustdata experience (both founders)", "value": "no overlap"}]}

    # --- Output ----------------------------------------------------------
    def whisper(self, claim: dict, verdict: dict) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=config.OPENAI_WHISPER_MODEL, max_tokens=60,
                messages=[{"role": "system", "content": WHISPER_PROMPT},
                          {"role": "user", "content": f"Claim: {claim['statement']}\nData: {verdict['finding']}"}])
            return (resp.choices[0].message.content or "").strip() or verdict["finding"]
        except Exception:  # never lose a contradiction because the phrasing call failed
            return verdict["finding"]

    def record(self, sid: str, utterance_id: str, claim: dict, verdict: dict, whisper: str | None, cypher: list) -> str:
        cid = uuid.uuid4().hex[:12]
        self.db.query("""
            MATCH (u:Utterance {id: $uid})
            CREATE (u)-[:MADE_CLAIM]->(c:Claim {id: $cid, session: $sid, type: $type, statement: $statement,
                status: $status, finding: $finding, whisper: $whisper, evidence: $evidence, cypher: $cypher,
                target: $target, at: datetime()})""", {"target": verdict.get("target"),
            "uid": utterance_id, "cid": cid, "sid": sid, "type": claim["type"], "statement": claim["statement"],
            "status": verdict["status"], "finding": verdict["finding"], "whisper": whisper,
            "evidence": json.dumps(verdict["evidence"]), "cypher": json.dumps(cypher)})
        return cid

    def demo_lines(self, card: dict) -> list[str]:
        resp = self.llm.chat.completions.create(
            model=config.OPENAI_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": DEMO_LINES_PROMPT},
                      {"role": "user", "content": json.dumps(card, ensure_ascii=False, default=str)}])
        return json.loads(resp.choices[0].message.content or "{}").get("lines", [])[:8]

    def memo(self, record: dict) -> str:
        resp = self.llm.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "system", "content": MEMO_PROMPT},
                      {"role": "user", "content": json.dumps(record, ensure_ascii=False, default=str)}])
        return (resp.choices[0].message.content or "").strip()
