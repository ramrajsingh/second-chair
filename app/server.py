"""Second Chair server: two call channels in, whispers + live dossier out, memo at the end.

    uvicorn app.server:app --port 8010
"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, dossier
from app.claims import ClaimEngine
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
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/config")
def api_config():
    return {"live_data": bool(config.CRUSTDATA_API_KEY),
            "sample": {"company_domain": "nimbusfreight.example",
                       "person_url": "https://www.linkedin.com/in/arjun-mehta-nimbus-demo"}}


@app.get("/api/demo_script")
def demo_script():
    return json.loads((config.ROOT / "data" / "demo_script.json").read_text())


class Prepare(BaseModel):
    company_domain: str
    person_url: str | None = None


@app.post("/api/session/prepare")
def prepare(req: Prepare):
    """Pre-call: fetch Crustdata into Neo4j and open a session. No API calls happen during the call."""
    t0 = time.time()
    try:
        keys = dossier.build(db, req.company_domain.strip(), (req.person_url or "").strip() or None)
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
        RETURN c.statement AS claim, c.finding AS finding, toString(date(s.started_at)) AS call_date
        ORDER BY s.started_at DESC LIMIT 5""", {"sid": sid, "pid": keys["person_pid"], "ckey": keys["company_key"]})
    names = [card["company"]["name"]] + [f["name"] for f in card["founders"]] + \
            [i.split(" (")[0] for r in card["rounds"] for i in r["investors"]]
    sessions[sid] = {**keys, "seq": 0, "recent": [], "keyterms": sorted(set(n for n in names if n)),
                     "context": f"Diligence call with {card['person']['name'] if card['person'] else 'a founder'} "
                                f"of {card['company']['name']}."}
    return {"session_id": sid, "dossier": card, "open_items": open_items, "graph": graph(sid)["graph"],
            "live_data": bool(config.CRUSTDATA_API_KEY), "ms": int((time.time() - t0) * 1000)}


@app.post("/api/session/{sid}/utterance")
def utterance(sid: str, channel: str = Form(...), audio: UploadFile | None = File(None),
              text: str | None = Form(None), speak: bool = Form(False)):
    """One chunk of speech on a channel. Channels are separate streams, so the speaker is known exactly."""
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
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
    context = s["context"] + " Recent: " + " | ".join(s["recent"][-4:])
    s["recent"].append(f"{channel}: {heard.english}")

    results = []
    if channel == "subject":
        t = time.time()
        claims = engine.extract(heard.english, context)
        timing["extract"] = int((time.time() - t) * 1000)
        t = time.time()

        def check(c):
            cypher: list = []
            verdict = engine.verify(c, s["company_key"], s["person_pid"], cypher)
            whisper = engine.whisper(c, verdict) if verdict["status"] == "contradicted" else None
            audio_chunks = voice.speak(whisper, "en-IN")[0] if whisper else []
            cid = engine.record(sid, uid, c, verdict, whisper, cypher)
            return {"id": cid, **c, **verdict, "whisper": whisper, "whisper_audio": audio_chunks, "cypher": cypher}

        results = list(POOL.map(check, claims))
        timing["verify+whisper"] = int((time.time() - t) * 1000)
    return {"utterance": {"id": uid, "channel": channel, "text": heard.english, "original": heard.original,
                          "language": heard.language},
            "claims": results, "subject_audio": subject_audio.result() if subject_audio else [],
            "timing_ms": timing}


@app.get("/api/session/{sid}/graph")
def graph(sid: str):
    """Dossier + claims as nodes and edges for the live graph view."""
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
    rows = db.query("""
        MATCH (c:Company {key: $ckey})
        OPTIONAL MATCH (f:Person)-[:FOUNDED]->(c)
        OPTIONAL MATCH (f)-[w:WORKED_AT]->(o:Company)
        OPTIONAL MATCH (c)-[:RAISED]->(r:FundingRound)<-[inv:INVESTED_IN]-(i:Investor)
        RETURN c, collect(DISTINCT f) AS founders, collect(DISTINCT [f, w.title, o]) AS jobs,
               collect(DISTINCT [r, i, inv.lead]) AS funding""", {"ckey": s["company_key"]})[0]
    claims = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(:Utterance)-[:MADE_CLAIM]->(cl:Claim)
        RETURN cl.id AS id, cl.statement AS statement, cl.status AS status, cl.type AS type""", {"sid": sid})
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
    target = {"headcount": cnode, "growth": cnode, "funding_total": cnode, "hiring": cnode}
    for cl in claims:
        cn = node(f"cl:{cl['id']}", cl["statement"][:40], f"claim_{cl['status']}")
        anchor = target.get(cl["type"]) or (f"p:{s['person_pid']}" if s["person_pid"] else cnode)
        edges.append({"from": cn, "to": anchor, "label": cl["status"].upper(), "dashes": True})
    return {"graph": {"nodes": list(nodes.values()), "edges": edges}}


@app.post("/api/session/{sid}/end")
def end(sid: str):
    """The service deliverable: a memo with a source on every claim, saved on the session."""
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
    card = dossier.summary(db, s["company_key"], s["person_pid"])
    transcript = db.query("""
        MATCH (:Session {id: $sid})-[:HAS_UTTERANCE]->(u:Utterance)
        OPTIONAL MATCH (u)-[:MADE_CLAIM]->(c:Claim)
        WITH u, collect(c {.statement, .status, .finding, .evidence}) AS claims ORDER BY u.seq
        RETURN u.speaker AS speaker, u.text AS text, claims""", {"sid": sid})
    memo = engine.memo({"dossier": card, "transcript": transcript})
    db.query("MATCH (s:Session {id: $sid}) SET s.memo = $memo, s.ended_at = datetime()", {"sid": sid, "memo": memo})
    counts = {}
    for u in transcript:
        for c in u["claims"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"memo": memo, "counts": counts}
