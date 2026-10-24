"""Build the pre-call dossier: Crustdata person + company data loaded into Neo4j.

    (:Person)-[:WORKED_AT {title, start, end, current}]->(:Company)
    (:Person)-[:FOUNDED]->(:Company)
    (:Person)-[:STUDIED_AT {degree, field}]->(:School)
    (:Company)-[:HEADCOUNT]->(:HeadcountPoint {date, count})
    (:Company)-[:RAISED]->(:FundingRound {date, amount_usd, round_type})<-[:INVESTED_IN {lead}]-(:Investor)

Everything is fetched before the call, so live claim checks only read the graph (no API calls in
the hot path).
"""

from app import crustdata
from app.neo import Neo, company_key


def _first(v):
    return (v[0] if v else None) if isinstance(v, list) else v


def _domain(v) -> str | None:
    v = _first(v)
    return v.lower().split("://")[-1].removeprefix("www.").split("/")[0].split("?")[0] if v else None


def _list(v) -> list[str]:
    """Investor lists arrive as arrays or as comma-separated text depending on the endpoint."""
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [x for x in (v or []) if x]


def _norm_job(j: dict, current: bool) -> dict:
    # Person enrichment uses name/title/start_date; founder profiles inside company enrichment use
    # employer_name/employee_title/employee_start_date. Dates are trimmed to YYYY-MM-DD.
    day = lambda d: d[:10] if d else None
    return {"name": j.get("name") or j.get("employer_name"),
            "domain": _domain(j.get("company_website_domain") or j.get("employer_website") or j.get("company_website")),
            "title": j.get("title") or j.get("employee_title") or "",
            "start": day(j.get("start_date") or j.get("employee_start_date")),
            "end": day(j.get("end_date") or j.get("employee_end_date")),
            "seniority": j.get("seniority_level"), "current": current}


def _jobs(profile: dict) -> list[dict]:
    details = ((profile or {}).get("experience") or {}).get("employment_details") or {}
    return [_norm_job(j, True) for j in details.get("current") or []] + \
           [_norm_job(j, False) for j in details.get("past") or []]


def _person_id(profile: dict) -> str:
    return str(profile.get("crustdata_person_id") or profile["basic_profile"]["name"])


def load_person(db: Neo, profile: dict, founded_key: str | None = None) -> str:
    pid = _person_id(profile)
    basic = profile.get("basic_profile") or {}
    db.query("""
        MERGE (p:Person {pid: $pid})
        SET p.name = $name, p.headline = $headline, p.current_title = $title, p.profile_url = $url,
            p.skills = $skills""", {
        "pid": pid, "name": basic.get("name"), "headline": basic.get("headline"),
        "title": basic.get("current_title"), "url": (profile.get("professional_network") or {}).get("profile_url"),
        "skills": ((profile.get("skills") or {}).get("professional_network_skills") or [])})
    for j in _jobs(profile):
        db.query("""
            MATCH (p:Person {pid: $pid})
            MERGE (c:Company {key: $ckey}) ON CREATE SET c.name = $cname, c.domain = $domain
            MERGE (p)-[w:WORKED_AT {title: $title, start: $start}]->(c)
            SET w.end = $end, w.current = $current, w.seniority = $seniority""", {
            "pid": pid, "ckey": company_key(j["name"], j["domain"]), "cname": j["name"], "domain": j["domain"],
            "title": j["title"], "start": j["start"] or "", "end": j["end"], "current": j["current"],
            "seniority": j["seniority"]})
    for s in ((profile.get("education") or {}).get("schools") or []):
        db.query("""
            MATCH (p:Person {pid: $pid}) MERGE (s:School {name: $school})
            MERGE (p)-[r:STUDIED_AT]->(s) SET r.degree = $degree, r.field = $field,
                r.start_year = $sy, r.end_year = $ey""", {
            "pid": pid, "school": s.get("school"), "degree": s.get("degree"), "field": s.get("field_of_study"),
            "sy": s.get("start_year"), "ey": s.get("end_year")})
    if founded_key:
        db.query("MATCH (p:Person {pid: $pid}), (c:Company {key: $ckey}) MERGE (p)-[:FOUNDED]->(c)",
                 {"pid": pid, "ckey": founded_key})
    return pid


def load_company(db: Neo, data: dict) -> str:
    basic = data.get("basic_info") or {}
    ckey = company_key(basic.get("name"), _domain(basic.get("primary_domain")))
    head = data.get("headcount") or {}
    fund = data.get("funding") or {}
    hiring = data.get("hiring") or {}
    db.query("""
        MERGE (c:Company {key: $ckey})
        SET c.name = $name, c.domain = $domain, c.description = $desc, c.year_founded = $founded,
            c.headcount = $total, c.growth_yoy_pct = $yoy, c.growth_6m_pct = $six,
            c.total_funding_usd = $funding, c.last_round_type = $last_round,
            c.openings = $openings, c.crustdata_id = $cid""", {
        "ckey": ckey, "name": basic.get("name"), "domain": basic.get("primary_domain"),
        "desc": basic.get("description"), "founded": basic.get("year_founded"), "total": head.get("total"),
        "yoy": (head.get("growth_percent") or {}).get("yoy"),
        "six": (head.get("growth_percent") or {}).get("six_months"),
        "funding": fund.get("total_investment_usd"), "last_round": fund.get("last_round_type"),
        "openings": hiring.get("openings_count"), "cid": data.get("crustdata_company_id")})
    db.query("""
        MATCH (c:Company {key: $ckey})
        UNWIND $points AS pt
        MERGE (h:HeadcountPoint {company: $ckey, date: date(pt.date)})
        SET h.count = pt.employee_count
        MERGE (c)-[:HEADCOUNT]->(h)""", {"ckey": ckey, "points": head.get("timeseries") or []})
    for m in fund.get("milestones") or []:
        date = (m.get("date") or m.get("funding_date") or m.get("announced_date") or "")[:10]
        if not date:
            continue
        db.query("""
            MATCH (c:Company {key: $ckey})
            MERGE (f:FundingRound {company: $ckey, date: date($date)})
            SET f.amount_usd = $amount, f.round_type = $round
            MERGE (c)-[:RAISED]->(f)
            WITH f UNWIND $investors AS inv
            MERGE (i:Investor {name: inv})
            MERGE (i)-[r:INVESTED_IN]->(f) SET r.lead = inv IN $leads""", {
            "ckey": ckey, "date": date, "amount": m.get("amount_usd"), "round": m.get("round_type"),
            "investors": _list(m.get("investors")), "leads": _list(m.get("lead_investors"))})
    for founder in ((data.get("people") or {}).get("founders") or []):
        load_person(db, founder, founded_key=ckey)
    return ckey


def build(db: Neo, company_domain: str, founder_urls: list[str] | None = None) -> dict:
    """Fetch from Crustdata and load into the graph. The first founder URL is the person on the call;
    all given founders are linked with FOUNDED so co-founder overlap checks work for any team."""
    company = crustdata.enrich_company(company_domain)
    if not company:
        raise ValueError(f"Crustdata has no company for {company_domain!r}")
    ckey = load_company(db, company)
    pids = [load_person(db, p, founded_key=ckey) for p in crustdata.enrich_people(founder_urls)] \
        if founder_urls else []
    if not pids:  # default to the founders Crustdata lists for the company
        founders = (company.get("people") or {}).get("founders") or []
        pids = [_person_id(f) for f in founders]
    return {"company_key": ckey, "person_pid": pids[0] if pids else None}


def summary(db: Neo, ckey: str, pid: str | None) -> dict:
    """The dossier card shown before and during the call."""
    company = db.query("""
        MATCH (c:Company {key: $ckey})
        OPTIONAL MATCH (c)-[:HEADCOUNT]->(h:HeadcountPoint)
        WITH c, h ORDER BY h.date
        WITH c, collect({date: toString(h.date), count: h.count}) AS series
        OPTIONAL MATCH (c)-[:RAISED]->(f:FundingRound)<-[r:INVESTED_IN]-(i:Investor)
        WITH c, series, f, collect(CASE WHEN r.lead THEN i.name + ' (lead)' ELSE i.name END) AS investors
        ORDER BY f.date
        RETURN c {.name, .domain, .description, .headcount, .growth_yoy_pct, .total_funding_usd, .openings} AS company,
               series, collect(CASE WHEN f IS NULL THEN null ELSE
                 {date: toString(f.date), round: f.round_type, amount_usd: f.amount_usd, investors: investors} END) AS rounds""",
                       {"ckey": ckey})[0]
    person = db.query("""
        MATCH (p:Person {pid: $pid})
        OPTIONAL MATCH (p)-[w:WORKED_AT]->(c:Company)
        WITH p, w, c ORDER BY w.start DESC
        RETURN p {.name, .headline, .profile_url} AS person,
               collect({company: c.name, title: w.title, start: w.start, end: w.end, current: w.current}) AS roles""",
                      {"pid": pid}) if pid else []
    founders = db.query("MATCH (p:Person)-[:FOUNDED]->(:Company {key: $ckey}) RETURN p.name AS name, p.current_title AS title",
                        {"ckey": ckey})
    return {**company, "person": person[0] if person else None, "founders": founders}
