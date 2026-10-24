# Second Chair

*Every diligence call deserves a second chair: one who's read the founder's whole history and checks every claim in real time.*

Second Chair is a voice copilot for founder diligence calls. It listens to both sides of a call, checks every claim the founder makes against live company and people data from **Crustdata** stored in a **Neo4j** graph, whispers a sharp follow-up to the interviewer when a claim doesn't hold up, and writes a diligence memo with a source on every claim when the call ends.

Built for **ContextCon Build: Services, Replaced** (Crustdata).

```
interviewer mic ─┐                                     ┌─► 🎧 whisper (Bulbul TTS) on contradictions
                 ├─► Sarvam STT (per channel) ─► claim spotting (LLM) ─► Cypher checks vs dossier graph ─┤
founder (tab) ───┘                                                                                      └─► claim cards + live graph
before the call: Crustdata person + company enrichment ─► Neo4j        after the call: cited memo ─► Neo4j
```

## Graph model (Neo4j)

```
(:Person)-[:WORKED_AT {title, start, end, current}]->(:Company)
(:Person)-[:FOUNDED]->(:Company)
(:Person)-[:STUDIED_AT {degree, field}]->(:School)
(:Company)-[:HEADCOUNT]->(:HeadcountPoint {date, count})
(:Company)-[:RAISED]->(:FundingRound {date, amount_usd, round_type})<-[:INVESTED_IN {lead}]-(:Investor)
(:Session)-[:ABOUT]->(:Person | :Company)
(:Session)-[:HAS_UTTERANCE]->(:Utterance {speaker, text, original, language})-[:MADE_CLAIM]->(:Claim {status, finding, evidence, cypher})
```

The dossier is fetched **before** the call (Crustdata's default limit is 15 requests/minute), so live checks are graph reads only. Contradictions from earlier sessions with the same founder are shown at the start of the next call.

## Claim checks (each one Cypher query)

| Claim | Checked against |
|---|---|
| Headcount ("we're 200 people") | Latest `HeadcountPoint` (±20%) |
| Growth ("we tripled this year") | Headcount N months ago → now |
| Role ("VP Eng at Flipkart 2018–21") | `WORKED_AT` title seniority and dates |
| Funding total / round | `total_funding_usd`, `FundingRound` amounts |
| Investor ("Peak Harbor led our A") | `Investor`-[:INVESTED_IN]->`FundingRound` |
| Hiring ("hiring 30 engineers") | Open roles and 6-month headcount change |
| Shared history ("my co-founder and I worked together") | Multi-hop: both founders `WORKED_AT` the same company with overlapping dates |

## Setup

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env        # Neo4j, Sarvam, OpenAI; Crustdata optional
.venv/bin/uvicorn app.server:app --port 8010
```

Open http://localhost:8010, enter a **company domain** and the **founders' profile URLs** (comma-separated; the first is the person on the call; leave empty to use the founders Crustdata lists), then click **Build dossier & start call**. Without `CRUSTDATA_API_KEY` it uses `data/sample_dossier.json` (a fictional founder and company).

**Demo on any company:** after the dossier loads, **▶ Next scripted founder line** plays lines generated from that company's real data: some accurate, some inflated. You see both verified and contradicted verdicts without a second person on the call.

Feed the call in any combination:
- **🎙 Mic:** the interviewer's microphone.
- **🖥 Tab audio:** share the video-call tab with "Share tab audio" ticked, for the founder's side.
- **Typed lines** on either channel.
- **▶ Next scripted founder line:** demo lines voiced in a male voice.

**End call → memo** writes the cited memo and saves it to the session.

Reset between demos: `python -m scripts.reset`
