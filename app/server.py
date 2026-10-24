"""Second Chair server: two call channels in, whispers + live dossier out, memo at the end.

    uvicorn app.server:app --port 8010
"""

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, crustdata, dossier
from app.claims import ClaimEngine, signature
from app.crustdata import CrustdataError
from app.neo import GraphError, Neo, setup_schema
from app.voice import Voice

log = logging.getLogger("copilot")
app = FastAPI(title="Second Chair")
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")

db = Neo()
setup_schema(db)
voice = Voice()
engine = ClaimEngine(db)
POOL = ThreadPoolExecutor(max_workers=12)
sessions: dict[str, dict] = {}


@app.get("/")
def index():
    # No caching: the page changes often and a stale copy silently runs old code.
    return FileResponse(config.ROOT / "static" / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/config")
def api_config():
    return {"live_data": bool(config.CRUSTDATA_API_KEY),
            "sample": {"company_domain": "nimbusfreight.example",
                       "founder_urls": "https://www.linkedin.com/in/arjun-mehta-nimbus-demo"}}


@app.get("/api/session/{sid}/demo_lines")
def demo_lines(sid: str):
    """Founder lines for a scripted demo: the hand-written script for the sample company, otherwise
    generated from the real dossier (a mix of accurate and inflated claims), so any company can be demoed."""
    s = _session(sid)
    if "demo_lines" not in s:
        if s["company_key"] == "nimbusfreight.example":
            path = config.ROOT / "data" / "demo_script.json"
        else:
            # Generated once per company, then saved so they can be edited: data/scripts/<domain>.json
            path = config.ROOT / "data" / "scripts" / f"{re.sub(r'[^a-z0-9.-]', '_', s['company_key'])}.json"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                lines = engine.demo_lines(dossier.summary(db, s["company_key"], s["person_pid"]))
                path.write_text(json.dumps({"_note": "Candidate lines for '▶ Next scripted candidate line'. Edit, reorder or add; "
                                                     "the next call uses this file.", "lines": lines}, indent=2, ensure_ascii=False))
        s["demo_lines"] = json.loads(path.read_text())["lines"]
        s["demo_file"] = str(path.relative_to(config.ROOT))
    return {"lines": s["demo_lines"], "file": s.get("demo_file")}


class Prepare(BaseModel):
    company_domain: str
    founder_urls: str | list[str] | None = None  # comma/newline separated; the first is on the call


def _urls(v) -> list[str]:
    items = v if isinstance(v, list) else re.split(r"[,\s]+", v or "")
    return [u.strip() for u in items if u.strip().startswith("http")]


@app.post("/api/session/prepare")
def prepare(req: Prepare):
    """Pre-call: fetch Crustdata into Neo4j and open a session. No API calls happen during the call."""
    t0 = time.time()
    try:
        domain = re.sub(r"^https?://(www\.)?", "", req.company_domain.strip()).strip("/")
        keys = dossier.build(db, domain, _urls(req.founder_urls))
        card = dossier.summary(db, keys["company_key"], keys["person_pid"])
    except (CrustdataError, GraphError, ValueError) as e:
        raise HTTPException(502, str(e))
    sid = uuid.uuid4().hex[:12]
    db.query("""
        MATCH (c:Company {key: $ckey}) OPTIONAL MATCH (p:Person {pid: $pid})
        CREATE (s:Session {id: $sid, started_at: datetime(), company: $ckey, person: $pid})
        MERGE (s)-[:ABOUT]->(c)
        FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (s)-[:ABOUT]->(p))""",
             {"sid": sid, "ckey": keys["company_key"], "pid": keys["person_pid"]})
    # Follow-ups across calls: contradictions from earlier sessions with this person or company.
    open_items = db.query("""
        MATCH (s:Session)-[:HAS_UTTERANCE]->(:Utterance)-[:MADE_CLAIM]->(c:Claim {status: 'contradicted'})
        WHERE s.id <> $sid AND (s.person = $pid OR s.company = $ckey)
        WITH c.finding AS finding, collect(c.statement)[0] AS claim, max(s.started_at) AS last
        RETURN claim, finding, toString(date(last)) AS call_date ORDER BY last DESC LIMIT 5""", {"sid": sid, "pid": keys["person_pid"], "ckey": keys["company_key"]})
    sessions[sid] = _new_state(keys, card)
    return {"session_id": sid, "dossier": card, "open_items": open_items, "graph": graph(sid)["graph"],
            "live_data": bool(config.CRUSTDATA_API_KEY), "ms": int((time.time() - t0) * 1000)}


def _new_state(keys: dict, card: dict) -> dict:
    """In-memory working state for a call: what claims can be checked against, keyterms, de-dup sets."""
    names = [card["company"]["name"]] + [f["name"] for f in card["founders"]] + \
            [i.split(" (")[0] for r in card["rounds"] for i in r["investors"]]
    founder_ids = [r["pid"] for r in db.query(
        "MATCH (p:Person)-[:FOUNDED]->(:Company {key: $ckey}) RETURN p.pid AS pid", {"ckey": keys["company_key"]})]
    return {**keys, "seq": 0, "recent": [], "seen": set(), "suggested": set(),
            # Everything this call's claims can be checked against; grows as entities are added.
            "scope": {"companies": {keys["company_key"]: card["company"]["name"]},
                      "people": founder_ids + ([keys["person_pid"]] if keys["person_pid"] else [])},
            "added": [],
            "keyterms": sorted(set(n for n in names if n)),
            "context": f"Diligence call with {card['person']['person']['name'] if card['person'] else 'a candidate'} "
                       f"of {card['company']['name']}."}


def _session(sid: str) -> dict:
    """The call's state, restored from Neo4j if the server restarted mid-call (every session is persisted)."""
    if sid in sessions:
        return sessions[sid]
    rows = db.query("""
        MATCH (s:Session {id: $sid})
        OPTIONAL MATCH (s)-[:HAS_UTTERANCE]->(u:Utterance)
        WITH s, max(u.seq) AS seq
        OPTIONAL MATCH (s)-[:ADDED]->(e)
        RETURN s.company AS company_key, s.person AS person_pid, coalesce(seq, 0) AS seq,
               collect(CASE WHEN e:Company THEN {kind: 'company', id: e.key, name: e.name}
                            WHEN e:Person THEN {kind: 'person', id: e.pid, name: e.name} END) AS added""", {"sid": sid})
    if not rows:
        raise HTTPException(404, "Unknown session: click 'Build dossier & start call' to start a new call.")
    r = rows[0]
    keys = {"company_key": r["company_key"], "person_pid": r["person_pid"]}
    state = _new_state(keys, dossier.summary(db, keys["company_key"], keys["person_pid"]))
    state["seq"] = r["seq"]
    for a in r["added"]:
        if a:
            state["added"].append(a)
            if a["kind"] == "company":
                state["scope"]["companies"][a["id"]] = a["name"]
            else:
                state["scope"]["people"].append(a["id"])
    sessions[sid] = state
    return state


@app.post("/api/session/{sid}/utterance")
def utterance(sid: str, channel: str = Form(...), audio: UploadFile | None = File(None),
              text: str | None = Form(None), speak: bool = Form(False)):
    """One chunk of speech on a channel. Channels are separate streams, so the speaker is known exactly."""
    s = _session(sid)
    if channel not in ("interviewer", "subject"):
        raise HTTPException(400, "channel must be interviewer or subject")
    timing, t = {}, time.time()
    if audio is not None:
        data = audio.file.read()
        codec = (audio.content_type or "audio/webm").split("/")[-1].split(";")[0]
        heard = voice.transcribe(data, audio.filename or f"chunk.{codec}", codec, s["keyterms"])
    elif text:
        heard = voice.from_text(text)
    else:
        raise HTTPException(400, "Send audio or text")
    timing["stt"] = int((time.time() - t) * 1000)
    if not heard.original.strip():
        return {"utterance": None}
    # Typed subject lines can be voiced, so a scripted demo sounds like a real second speaker.
    subject_audio = POOL.submit(voice.speak_caller, heard.original, heard.language, "male") \
        if speak and audio is None else None

    s["seq"] += 1
    uid = f"{sid}-{s['seq']}"
    db.query("""
        MATCH (s:Session {id: $sid})
        CREATE (s)-[:HAS_UTTERANCE]->(:Utterance {id: $uid, seq: $seq, speaker: $channel, text: $text,
               original: $original, language: $lang, at: datetime()})""",
             {"sid": sid, "uid": uid, "seq": s["seq"], "channel": channel, "text": heard.english,
              "original": heard.original, "lang": heard.language})
    context = s["context"] + " Previous line: " + (s["recent"][-1] if s["recent"] else "none")
    s["recent"].append(f"{channel}: {heard.english}")

    results, flags = [], []
    if channel == "interviewer":
        # The latest interviewer line is the open question; replies are judged against it until answered.
        s["open_q"] = {"uid": uid, "text": heard.english, "answer": ""}
    if channel == "subject":
        t = time.time()
        q = s.get("open_q")
        evasion_f = None
        if q:
            q["answer"] = (q["answer"] + " " + heard.english).strip()
            evasion_f = POOL.submit(engine.evasion, q["text"], q["answer"])  # runs alongside extraction
        claims = []
        extracted = engine.extract(heard.english, context)
        for c in extracted["claims"]:
            if c["type"] == "funding_round" and c["number"] is None and not c["investor"] and not c["round_type"]:
                continue  # a round mentioned with nothing checkable
            if signature(c) not in s["seen"]:  # a claim repeated later in the call is checked once
                s["seen"].add(signature(c))
                claims.append(c)
        timing["extract"] = int((time.time() - t) * 1000)
        t = time.time()

        results = list(POOL.map(lambda c: _check_claim(s, sid, uid, c), claims))
        timing["verify"] = int((time.time() - t) * 1000)
        suggestions = _suggestions(s, extracted["mentions"])
        if evasion_f:
            ev = evasion_f.result()
            if not ev["is_question"] or ev["verdict"] == "answered":
                s["open_q"] = None
            else:  # flagged once per question
                s["open_q"] = None
                detail = f"{'Evaded' if ev['verdict'] == 'evaded' else 'Partly answered'}: “{q['text']}”. Missing: {ev['missing']}."
                fid = engine.record_flag(sid, "evasion", uid, q["uid"], detail, ev["follow_up"])
                s.setdefault("whispers", {})[fid] = POOL.submit(_speak, fid, f"Not answered: {ev['missing']}. Ask: {ev['follow_up']}")
                flags.append({"id": fid, "kind": "evasion", "verdict": ev["verdict"], "question": q["text"],
                              "missing": ev["missing"], "follow_up": ev["follow_up"], "detail": detail,
                              "whisper_pending": True})
            timing["evasion"] = "parallel"
    else:
        suggestions = []
    return {"utterance": {"id": uid, "channel": channel, "text": heard.english, "original": heard.original,
                          "language": heard.language},
            "claims": results, "flags": flags, "suggestions": suggestions, "subject_audio": subject_audio.result() if subject_audio else [],
            "timing_ms": timing}


def _check_claim(s: dict, sid: str, uid: str, c: dict) -> dict:
    cypher: list = []
    verdict = engine.verify(c, s["company_key"], s["person_pid"], cypher, s["scope"])
    cid = engine.record(sid, uid, c, verdict, None, cypher)
    shift = engine.find_shift(cid, cypher)
    if verdict["status"] == "contradicted":
        # The card goes back now; the spoken whisper is prepared in the background
        # and fetched by the page from /whisper/{cid}. A shifting answer makes it sharper.
        said = {**verdict, "finding": verdict["finding"] + (
            f" Also, {shift['when']} they said {shift['previous']:g}." if shift else "")}
        s.setdefault("whispers", {})[cid] = POOL.submit(_whisper, cid, c, said)
    elif shift:
        s.setdefault("whispers", {})[cid] = POOL.submit(
            _speak, cid, f"Shifting answer: {shift['when']} it was {shift['previous']:g}, now {c['number']:g}. Ask which is right.")
    elif verdict["status"] == "unverifiable" and verdict.get("target") is None and c["company"]:
        s.setdefault("pending", []).append((uid, c))  # re-checked when that company is added
    return {"id": cid, **c, **verdict, "shift": shift, "cypher": cypher,
            "whisper_pending": verdict["status"] == "contradicted" or bool(shift)}


def _known(s: dict, name: str) -> bool:
    n = name.lower()
    names = [v for v in s["scope"]["companies"].values()] + [a["name"] for a in s["added"]] + s["keyterms"]
    return any(x and (n in x.lower() or x.lower() in n) for x in names)


def _suggestions(s: dict, mentions: list[dict]) -> list[dict]:
    """Companies and people the founder named that aren't in the dossier yet, offered once each."""
    out = []
    for m in mentions:
        key = (m["kind"], m["name"].lower())
        if m["name"].strip() and key not in s["suggested"] and not _known(s, m["name"]):
            s["suggested"].add(key)
            out.append(m)
    return out


class AddEntity(BaseModel):
    value: str  # company domain, company name, or person profile URL


@app.post("/api/session/{sid}/dossier/add")
def add_to_dossier(sid: str, req: AddEntity):
    """Enrich a company or person mentioned on the call and add it to this call's dossier."""
    s = _session(sid)
    t0 = time.time()
    try:
        ent = dossier.add_entity(db, req.value)
    except (CrustdataError, GraphError, ValueError) as e:
        raise HTTPException(502, str(e))
    label = "Company" if ent["kind"] == "company" else "Person"
    match = "{key: $id}" if ent["kind"] == "company" else "{pid: $id}"
    db.query(f"MATCH (s:Session {{id: $sid}}), (e:{label} {match}) MERGE (s)-[r:ADDED]->(e) SET r.at = datetime()",
             {"sid": sid, "id": ent["id"]})
    if ent["kind"] == "company":
        s["scope"]["companies"][ent["id"]] = ent["name"]
        info = db.query("MATCH (c:Company {key: $id}) RETURN c.headcount AS headcount, c.total_funding_usd AS funding,"
                        " c.growth_yoy_pct AS yoy", {"id": ent["id"]})[0]
    else:
        s["scope"]["people"].append(ent["id"])
        info = {"roles": db.query("""MATCH (:Person {pid: $id})-[w:WORKED_AT]->(c:Company)
            RETURN c.key AS key, c.name AS company, w.title AS title, w.start AS start, w.end AS end
            ORDER BY w.start DESC LIMIT 6""",
                                  {"id": ent["id"]})}
    if ent["name"]:
        s["keyterms"] = sorted(set(s["keyterms"] + [ent["name"]]))
    entry = {**ent, **info}
    s["added"].append(entry)
    # Claims made before this company was in the dossier get checked now.
    ready = [(uid, c) for uid, c in s.get("pending", [])
             if ent["kind"] == "company" and engine._resolve_company(c["company"], {ent["id"]: ent["name"]})]
    s["pending"] = [p for p in s.get("pending", []) if p not in ready]
    rechecked = list(POOL.map(lambda p: _check_claim(s, sid, p[0], p[1]), ready))
    return {"added": entry, "rechecked": rechecked, "graph": graph(sid)["graph"],
            "ms": int((time.time() - t0) * 1000)}


def _speak(wid: str, text: str) -> dict:
    return {"text": text, "audio": voice.speak(text, "en-IN")[0]}


@app.get("/api/session/{sid}/node/{node_id:path}")
def node_detail(sid: str, node_id: str):
    """What the graph knows about a clicked node, and which actions make sense for it."""
    _session(sid)
    kind, _, ident = node_id.partition(":")
    if kind == "co":
        rows = db.query("""
            MATCH (c:Company {key: $key})
            OPTIONAL MATCH (p:Person)-[w:WORKED_AT]->(c)
            WITH c, collect({name: p.name, title: w.title, start: w.start, end: w.end}) AS people
            RETURN c.name AS name, c.domain AS domain, c.key AS key, c.headcount AS headcount,
                   c.total_funding_usd AS funding, c.growth_yoy_pct AS yoy, c.crustdata_id IS NOT NULL AS enriched,
                   people""", {"key": ident})
        if not rows:
            raise HTTPException(404, "Unknown company")
        return {"kind": "company", **rows[0]}
    if kind == "p":
        rows = db.query("""
            MATCH (p:Person {pid: $pid})
            OPTIONAL MATCH (p)-[w:WORKED_AT]->(c:Company)
            WITH p, w, c ORDER BY w.start DESC
            RETURN p.name AS name, p.headline AS headline, p.profile_url AS url,
                   collect({key: c.key, company: c.name, domain: c.domain, title: w.title, start: w.start, end: w.end}) AS roles""",
                        {"pid": ident})
        if not rows:
            raise HTTPException(404, "Unknown person")
        return {"kind": "person", **rows[0]}
    if kind == "i":
        return {"kind": "investor", "name": ident}
    return {"kind": "other"}


@app.get("/api/session/{sid}/overlap/{pid}")
def overlap(sid: str, pid: str):
    """What connects a person to the founder on the call: overlapping employers, GitHub orgs, skills, languages."""
    s = _session(sid)
    if not s["person_pid"]:
        raise HTTPException(404, "No candidate on this call")
    log: list = []
    row = db.query("""
        MATCH (a:Person {pid: $a}), (b:Person {pid: $b})
        OPTIONAL MATCH (a)-[wa:WORKED_AT]->(co:Company)<-[wb:WORKED_AT]-(b)
          WHERE wa.start <= coalesce(wb.end, '9999') AND wb.start <= coalesce(wa.end, '9999')
        WITH a, b, collect(DISTINCT co.name + ' (' + left(CASE WHEN wa.start > wb.start THEN wa.start ELSE wb.start END, 7) + ')') AS companies
        OPTIONAL MATCH (a)-[:HAS_GITHUB]->(:GitHubAccount)-[:MEMBER_OF]->(o:GitHubOrg)<-[:MEMBER_OF]-(:GitHubAccount)<-[:HAS_GITHUB]-(b)
        WITH a, b, companies, collect(DISTINCT o.name) AS orgs
        OPTIONAL MATCH (a)-[:HAS_SKILL]->(k:Skill)<-[:HAS_SKILL]-(b)
        WITH a, b, companies, orgs, collect(DISTINCT k.name) AS skills
        OPTIONAL MATCH (a)-[:HAS_GITHUB]->(:GitHubAccount)-[:OWNS]->(:Repo)-[:WRITTEN_IN]->(l:Language)
                       <-[:WRITTEN_IN]-(:Repo)<-[:OWNS]-(:GitHubAccount)<-[:HAS_GITHUB]-(b)
        RETURN a.name AS candidate, b.name AS person, companies, orgs, skills, collect(DISTINCT l.name) AS languages""",
                   {"a": s["person_pid"], "b": pid}, log, "overlap: employers, GitHub orgs, skills, languages")
    if not row:
        raise HTTPException(404, "Unknown person")
    return {**row[0], "cypher": log}


class PeopleSearch(BaseModel):
    company_key: str
    when: str = "any"           # current, past or any
    title: str | None = None
    limit: int = 10


@app.post("/api/session/{sid}/people_search")
def people_search(sid: str, req: PeopleSearch):
    """Crustdata people search scoped to a company in the graph, e.g. former engineering leads,
    for reference checks or to verify "I worked with X"."""
    _session(sid)
    co = db.query("MATCH (c:Company {key: $key}) RETURN c.name AS name, c.domain AS domain", {"key": req.company_key})
    if not co:
        raise HTTPException(404, "Unknown company")
    t0 = time.time()
    try:
        found = crustdata.search_people(domain=co[0]["domain"], company=co[0]["name"], when=req.when,
                                        title=(req.title or "").strip() or None, limit=max(1, min(req.limit, 25)))
    except CrustdataError as e:
        raise HTTPException(502, str(e))
    in_graph = {r["pid"] for r in db.query("MATCH (p:Person) WHERE p.pid IN $ids RETURN p.pid AS pid",
                                           {"ids": [f["pid"] for f in found]})}
    return {"company": co[0]["name"], "results": [{**f, "in_graph": f["pid"] in in_graph} for f in found],
            "ms": int((time.time() - t0) * 1000)}


def _whisper(cid: str, claim: dict, verdict: dict) -> dict:
    text = engine.whisper(claim, verdict)
    db.query("MATCH (c:Claim {id: $cid}) SET c.whisper = $text", {"cid": cid, "text": text})
    return {"text": text, "audio": voice.speak(text, "en-IN")[0]}


@app.get("/api/session/{sid}/whisper/{cid}")
def whisper(sid: str, cid: str):
    """Wait for a contradiction's whisper (text + speech), prepared right after the verdict."""
    fut = sessions.get(sid, {}).get("whispers", {}).get(cid)
    if not fut:
        raise HTTPException(404, "No whisper for this claim")
    return fut.result(timeout=30)


@app.get("/api/session/{sid}/graph")
def graph(sid: str):
    """Dossier + claims as nodes and edges for the live graph view."""
    s = _session(sid)
    rows = db.query("""
        MATCH (c:Company {key: $ckey})
        OPTIONAL MATCH (f:Person)-[:FOUNDED]->(c)
        OPTIONAL MATCH (f)-[w:WORKED_AT]->(o:Company)
        OPTIONAL MATCH (c)-[:RAISED]->(r:FundingRound)<-[inv:INVESTED_IN]-(i:Investor)
        RETURN c, collect(DISTINCT f) AS founders, collect(DISTINCT [f, w.title, o]) AS jobs,
               collect(DISTINCT [r, i, inv.lead]) AS funding""", {"ckey": s["company_key"]})[0]
    claims = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(:Utterance)-[:MADE_CLAIM]->(cl:Claim)
        RETURN cl.id AS id, cl.statement AS statement, cl.status AS status, cl.type AS type,
               cl.target AS target""", {"sid": sid})
    nodes, edges = {}, []

    def node(nid, label, kind, extra=None):
        nodes.setdefault(nid, {"id": nid, "label": label, "kind": kind, **(extra or {})})
        return nid

    co = rows["c"]
    cnode = node(f"co:{co['key']}", f"{co['name']}\n{co.get('headcount') or '?'} people", "company")
    for f in rows["founders"]:
        edges.append({"from": node(f"p:{f['pid']}", f["name"], "person"), "to": cnode, "label": "FOUNDED"})
    for f, title, o in rows["jobs"]:
        if f and o and o["key"] != co["key"]:
            edges.append({"from": f"p:{f['pid']}", "to": node(f"co:{o['key']}", o["name"], "past_company"),
                          "label": title or "WORKED_AT"})
    for r, i, lead in rows["funding"]:
        if r:
            rn = node(f"r:{r['round_type']}:{r['date']}", f"{(r['round_type'] or '').replace('_', ' ').title()}\n"
                      f"${(r['amount_usd'] or 0) / 1e6:.1f}M", "round")
            edges.append({"from": cnode, "to": rn, "label": "RAISED"})
            if i:
                edges.append({"from": node(f"i:{i['name']}", i["name"], "investor"), "to": rn,
                              "label": "LED" if lead else "INVESTED"})
    # Founders' GitHub: account, top repos by stars, and orgs (shared orgs show up as shared nodes).
    for g in db.query("""
        MATCH (f:Person)-[:FOUNDED]->(:Company {key: $ckey})
        MATCH (f)-[:HAS_GITHUB]->(gh:GitHubAccount)
        OPTIONAL MATCH (gh)-[:OWNS]->(r:Repo) WITH f, gh, r ORDER BY r.stars DESC
        WITH f, gh, collect(r)[..3] AS repos
        OPTIONAL MATCH (gh)-[:MEMBER_OF]->(o:GitHubOrg)
        RETURN f.pid AS pid, gh.login AS login, gh.total_stars AS stars, [x IN repos | x {.full_name, .stars}] AS repos,
               collect(o.name) AS orgs""", {"ckey": s["company_key"]}):
        gn = node(f"gh:{g['login']}", f"GitHub {g['login']}\n{g['stars'] or 0}★", "github")
        edges.append({"from": f"p:{g['pid']}", "to": gn, "label": "HAS_GITHUB"})
        for r in g["repos"]:
            edges.append({"from": gn, "to": node(f"repo:{r['full_name']}", f"{r['full_name'].split('/')[-1]}\n{r['stars'] or 0}★", "repo"), "label": "OWNS"})
        for o in g["orgs"][:5]:
            edges.append({"from": gn, "to": node(f"org:{o.lower()}", o, "github_org"), "label": "MEMBER_OF"})
    for a in s["added"]:
        if a["kind"] == "company":
            node(f"co:{a['id']}", f"{a['name']}\n{a.get('headcount') or '?'} people", "added_company")
        else:
            pn = node(f"p:{a['id']}", a["name"] or a["id"], "added_person")
            for r in a.get("roles", []):
                edges.append({"from": pn, "to": node(f"co:{r['key']}", r["company"], "past_company"),
                              "label": r["title"] or "WORKED_AT"})
    target = {"headcount": cnode, "growth": cnode, "funding_total": cnode, "hiring": cnode}
    for cl in claims:
        cn = node(f"cl:{cl['id']}", cl["statement"][:40], f"claim_{cl['status']}")
        anchor = (f"co:{cl['target']}" if cl.get("target") and cl["target"] != co["key"] and cl["type"] in target
                  else target.get(cl["type"])) or (f"p:{s['person_pid']}" if s["person_pid"] else cnode)
        edges.append({"from": cn, "to": anchor, "label": cl["status"].upper(), "dashes": True})
    return {"graph": {"nodes": list(nodes.values()), "edges": edges}}


@app.post("/api/session/{sid}/end")
def end(sid: str):
    """The service deliverable: a memo with a source on every claim, saved on the session."""
    s = _session(sid)
    card = dossier.summary(db, s["company_key"], s["person_pid"])
    transcript = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(u:Utterance)
        OPTIONAL MATCH (u)-[:MADE_CLAIM]->(c:Claim)
        WITH u, collect(c {.statement, .status, .finding, .evidence, .hedge}) AS claims ORDER BY u.seq
        RETURN u.speaker AS speaker, u.text AS text, claims""", {"sid": sid})
    evasions = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(a:Utterance)-[:RAISED]->(f:Flag {kind: 'evasion'})
        OPTIONAL MATCH (f)-[:ABOUT_QUESTION]->(q:Utterance)
        RETURN q.text AS question, a.text AS answer, f.detail AS detail, f.follow_up AS follow_up""", {"sid": sid})
    shifts = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(:Utterance)-[:MADE_CLAIM]->(cl:Claim)-[:SHIFTED_FROM]->(prev:Claim)
        RETURN cl.statement AS now, prev.statement AS before, prev.session = $sid AS same_call,
               toString(date(prev.at)) AS before_date""", {"sid": sid})
    memo = engine.memo({"dossier": card, "transcript": transcript, "evasions": evasions, "shifting_answers": shifts})
    db.query("MATCH (s:Session {id: $sid}) SET s.memo = $memo, s.ended_at = datetime()", {"sid": sid, "memo": memo})
    counts = {}
    for u in transcript:
        for c in u["claims"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
    if evasions:
        counts["evasions"] = len(evasions)
    if shifts:
        counts["shifting answers"] = len(shifts)
    return {"memo": memo, "counts": counts}
