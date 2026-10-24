"""Build the pre-call dossier: Crustdata person + company data loaded into Neo4j.

    (:Person)-[:WORKED_AT {title, start, end, current}]->(:Company)
    (:Person)-[:FOUNDED]->(:Company)
    (:Person)-[:STUDIED_AT {degree, field}]->(:School)
    (:Company)-[:HEADCOUNT]->(:HeadcountPoint {date, count})
    (:Company)-[:RAISED]->(:FundingRound {date, amount_usd, round_type})<-[:INVESTED_IN {lead}]-(:Investor)

Everything is fetched before the call, so live claim checks only read the graph (no API calls in
the hot path).
"""

import re
from concurrent.futures import ThreadPoolExecutor

from app import crustdata
from app.neo import Neo, company_key


_POOL = ThreadPoolExecutor(max_workers=8)


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


def _profile_url(profile: dict) -> str | None:
    return ((profile.get("professional_network") or {}).get("profile_url")
            or ((profile.get("social_handles") or {}).get("professional_network_identifier") or {}).get("profile_url"))


def _person_id(profile: dict) -> str:
    return str(profile.get("crustdata_person_id") or profile["basic_profile"]["name"])


def load_person(db: Neo, profile: dict, founded_key: str | None = None) -> str:
    pid = _person_id(profile)
    basic = profile.get("basic_profile") or {}
    net = profile.get("professional_network") or {}
    auth = (profile.get("assessment") or {}).get("authenticity") or {}
    db.query("""
        MERGE (p:Person {pid: $pid})
        SET p.name = $name, p.headline = coalesce($headline, p.headline), p.current_title = coalesce($title, p.current_title),
            p.profile_url = coalesce($url, p.profile_url),
            p.followers = coalesce($followers, p.followers), p.connections = coalesce($connections, p.connections),
            p.authenticity_verdict = coalesce($verdict, p.authenticity_verdict),
            p.authenticity_tier = coalesce($tier, p.authenticity_tier)""", {
        "pid": pid, "name": basic.get("name"), "headline": basic.get("headline"),
        "title": basic.get("current_title"),
        "url": net.get("profile_url")
               or ((profile.get("social_handles") or {}).get("professional_network_identifier") or {}).get("profile_url"),
        "followers": net.get("followers"), "connections": net.get("connections"),
        "verdict": auth.get("verdict"), "tier": auth.get("tier")})
    skills = [s for s in ((profile.get("skills") or {}).get("professional_network_skills") or []) if s]
    if skills:
        db.query("""MATCH (p:Person {pid: $pid}) UNWIND $skills AS s
            MERGE (k:Skill {key: toLower(trim(s))}) ON CREATE SET k.name = s
            MERGE (p)-[:HAS_SKILL]->(k)""", {"pid": pid, "skills": skills[:60]})
    for gh in profile.get("dev_platform_profiles") or []:
        load_github(db, pid, gh)
    # One round trip per list (each Aura round trip is ~100 ms).
    jobs = [{"ckey": company_key(j["name"], j["domain"]), "cname": j["name"], "domain": j["domain"],
             "title": j["title"], "start": j["start"] or "", "end": j["end"], "current": j["current"],
             "seniority": j["seniority"]} for j in _jobs(profile) if j["name"]]
    if jobs:
        db.query("""
            MATCH (p:Person {pid: $pid})
            UNWIND $jobs AS j
            MERGE (c:Company {key: j.ckey}) ON CREATE SET c.name = j.cname, c.domain = j.domain
            MERGE (p)-[w:WORKED_AT {title: j.title, start: j.start}]->(c)
            SET w.end = j.end, w.current = j.current, w.seniority = j.seniority""", {"pid": pid, "jobs": jobs})
    schools = [{"school": s.get("school"), "degree": s.get("degree"), "field": s.get("field_of_study"),
                "sy": s.get("start_year"), "ey": s.get("end_year")}
               for s in ((profile.get("education") or {}).get("schools") or []) if s.get("school")]
    if schools:
        db.query("""
            MATCH (p:Person {pid: $pid})
            UNWIND $schools AS e
            MERGE (s:School {name: e.school})
            MERGE (p)-[r:STUDIED_AT]->(s) SET r.degree = e.degree, r.field = e.field,
                r.start_year = e.sy, r.end_year = e.ey""", {"pid": pid, "schools": schools})
    if founded_key:
        db.query("MATCH (p:Person {pid: $pid}), (c:Company {key: $ckey}) MERGE (p)-[:FOUNDED]->(c)",
                 {"pid": pid, "ckey": founded_key})
    return pid


def load_github(db: Neo, pid: str, gh: dict):
    """(:Person)-[:HAS_GITHUB]->(:GitHubAccount)-[:OWNS]->(:Repo)-[:WRITTEN_IN]->(:Language), plus org memberships."""
    url = gh.get("profile_url")
    if not url:
        return
    db.query("""
        MATCH (p:Person {pid: $pid})
        MERGE (g:GitHubAccount {url: $url})
        SET g.login = $login, g.followers = $followers, g.following = $following,
            g.public_repos = $repos, g.total_stars = $stars, g.max_stars = $max_stars,
            g.languages = $languages, g.topics = $topics, g.bio = $bio
        MERGE (p)-[:HAS_GITHUB]->(g)""", {
        "pid": pid, "url": url, "login": url.rstrip("/").rsplit("/", 1)[-1], "followers": gh.get("followers"),
        "following": gh.get("following"), "repos": gh.get("public_repo_count") or gh.get("total_repos"),
        "stars": gh.get("total_stars"), "max_stars": gh.get("max_stars"), "bio": gh.get("bio"),
        "languages": gh.get("all_languages") or [], "topics": (gh.get("all_topics") or [])[:30]})
    repos = [r for r in gh.get("repos") or [] if r.get("full_name") and not r.get("is_fork")]
    db.query("""
        MATCH (g:GitHubAccount {url: $url})
        UNWIND $repos AS r
        MERGE (repo:Repo {full_name: r.full_name})
        SET repo.description = r.description, repo.stars = r.stars, repo.forks = r.forks_count,
            repo.language = r.primary_language, repo.topics = coalesce(r.topics, []), repo.archived = r.archived,
            repo.pushed_at = r.github_pushed_at
        MERGE (g)-[:OWNS]->(repo)
        FOREACH (_ IN CASE WHEN r.primary_language IS NULL THEN [] ELSE [1] END |
            MERGE (l:Language {name: r.primary_language}) MERGE (repo)-[:WRITTEN_IN]->(l))""",
             {"url": url, "repos": repos[:25]})
    orgs = []
    for o in gh.get("org_memberships") or []:
        name = o.get("login") or o.get("name") if isinstance(o, dict) else o
        if name:
            orgs.append(str(name))
    if orgs:
        db.query("""MATCH (g:GitHubAccount {url: $url}) UNWIND $orgs AS o
            MERGE (org:GitHubOrg {key: toLower(o)}) ON CREATE SET org.name = o
            MERGE (g)-[:MEMBER_OF]->(org)""", {"url": url, "orgs": orgs})


def load_company(db: Neo, data: dict, skip_founders: set[str] | None = None) -> str:
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
    # Founders not already loaded from richer person-enrichment profiles come from the company data.
    founders = [f for f in (data.get("people") or {}).get("founders") or []
                if _person_id(f) not in (skip_founders or set())]
    list(_POOL.map(lambda f: load_person(db, f, founded_key=ckey), founders))
    return ckey


def build(db: Neo, company_domain: str, founder_urls: list[str] | None = None) -> dict:
    """Fetch from Crustdata and load into the graph. The first founder URL is the person on the call;
    all given founders are linked with FOUNDED so co-founder overlap checks work for any team."""
    company = crustdata.enrich_company(company_domain)
    if not company:
        raise ValueError(f"Crustdata has no company for {company_domain!r}")
    founders = (company.get("people") or {}).get("founders") or []
    if not founder_urls and crustdata.config.CRUSTDATA_API_KEY:
        # Founder profiles inside company data have no skills/GitHub/authenticity; enrich them all in one call.
        founder_urls = [u for u in (_profile_url(f) for f in founders) if u]
    enriched = crustdata.enrich_people(founder_urls) if founder_urls else []
    ckey = load_company(db, company, skip_founders={_person_id(p) for p in enriched})
    pids = list(_POOL.map(lambda p: load_person(db, p, founded_key=ckey), enriched))
    if not pids:  # default to the founders Crustdata lists for the company
        pids = [_person_id(f) for f in founders]
    return {"company_key": ckey, "person_pid": pids[0] if pids else None}


def add_entity(db: Neo, value: str) -> dict:
    """Enrich something mentioned on the call and load it into the graph.
    A profile URL adds a person; a domain or URL adds a company; anything else is a company name."""
    v = value.strip()
    if not crustdata.config.CRUSTDATA_API_KEY:
        raise ValueError("Adding to the dossier needs CRUSTDATA_API_KEY; sample mode has one fictional company.")
    if "linkedin.com/in/" in v:
        people = crustdata.enrich_people([v])
        if not people:
            raise ValueError(f"Crustdata has no profile for {v}")
        pid = load_person(db, people[0])
        return {"kind": "person", "id": pid, "name": (people[0].get("basic_profile") or {}).get("name")}
    is_domain = bool(re.match(r"^(https?://)?[\w-]+(\.[\w-]+)+(/.*)?$", v)) and " " not in v
    data = crustdata.enrich_company(domain=_domain(v)) if is_domain else crustdata.enrich_company(name=v)
    if not data:
        raise ValueError(f"Crustdata has no company for {v!r}")
    ckey = load_company(db, data)
    return {"kind": "company", "id": ckey, "name": (data.get("basic_info") or {}).get("name")}


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
        WITH p, collect({company: c.name, title: w.title, start: w.start, end: w.end, current: w.current}) AS roles
        OPTIONAL MATCH (p)-[:HAS_SKILL]->(k:Skill)
        WITH p, roles, collect(k.name)[..15] AS skills
        OPTIONAL MATCH (p)-[:HAS_GITHUB]->(g:GitHubAccount)
        OPTIONAL MATCH (g)-[:OWNS]->(r:Repo)
        WITH p, roles, skills, g, r ORDER BY r.stars DESC
        WITH p, roles, skills, g, collect(r {.full_name, .stars, .language, .description})[..3] AS top_repos
        OPTIONAL MATCH (g)-[:MEMBER_OF]->(o:GitHubOrg)
        WITH p, roles, skills, g, top_repos, collect(o.name) AS orgs
        RETURN p {.name, .headline, .profile_url, .followers, .connections, .authenticity_verdict, .authenticity_tier} AS person,
               roles, skills,
               CASE WHEN g IS NULL THEN null ELSE g {.url, .login, .followers, .public_repos, .total_stars, .languages,
                    top_repos: top_repos, orgs: orgs} END AS github""",
                      {"pid": pid}) if pid else []
    founders = db.query("""MATCH (p:Person)-[:FOUNDED]->(:Company {key: $ckey})
        RETURN p.name AS name, p.current_title AS title, p.authenticity_verdict AS authenticity""", {"ckey": ckey})
    return {**company, "person": person[0] if person else None, "founders": founders}
