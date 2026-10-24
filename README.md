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
(:Session)-[:ADDED]->(:Person | :Company)          // added mid-call
(:Utterance)-[:RAISED]->(:Flag {kind: 'evasion'})-[:ABOUT_QUESTION]->(:Utterance)
(:Person {followers, connections, authenticity_verdict, authenticity_tier})-[:HAS_SKILL]->(:Skill)
(:Person)-[:HAS_GITHUB]->(:GitHubAccount {total_stars, languages})-[:OWNS]->(:Repo {stars})-[:WRITTEN_IN]->(:Language)
(:GitHubAccount)-[:MEMBER_OF]->(:GitHubOrg)
(:Claim {claimed, hedge})-[:SHIFTED_FROM]->(:Claim)
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
| Shared history ("my co-founder and I worked together") | Multi-hop: both founders `WORKED_AT` the same company with overlapping dates; without a company named, a shared GitHub org also counts |
| Open source ("my library has 2k stars") | Owned repo matched by key words → `Repo.stars`, or account total stars |
| Skill ("I've written Go for years") | `HAS_SKILL` or owned repos `WRITTEN_IN` that language; never "contradicted", since missing evidence isn't proof |
| GitHub org ("I'm a Kubernetes maintainer") | Public `MEMBER_OF` memberships (private ones can't be seen, so a miss is "unverifiable") |

## Setup

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env        # Neo4j, Sarvam, OpenAI; Crustdata optional
.venv/bin/uvicorn app.server:app --port 8010
```

Open http://localhost:8010, enter a **company domain** and the **founders' profile URLs** (comma-separated; the first is the person on the call; leave empty to use the founders Crustdata lists), then click **Build dossier & start call**. Without `CRUSTDATA_API_KEY` it uses `data/sample_dossier.json` (a fictional founder and company).

**Demo on any company:** after the dossier loads, **▶ Next scripted founder line** plays lines generated from that company's real data: some accurate, some inflated. They're saved to `data/scripts/<domain>.json` the first time, so you can edit, reorder or add lines; the next call uses your version. The sample company's lines are in `data/demo_script.json`. You see both verified and contradicted verdicts without a second person on the call.

Feed the call in any combination:
- **🎙 Mic:** the interviewer's microphone.
- **🖥 Tab audio:** share the video-call tab with "Share tab audio" ticked, for the founder's side.
- **Typed lines** on either channel.
- **▶ Next scripted founder line:** demo lines voiced in a male voice.

**Evasions and shifting answers:** beyond checking facts, Second Chair watches *how* the founder answers:
- **Unanswered questions:** each reply to an interviewer question is judged *answered / partly answered / evaded* (run alongside claim spotting, so no extra wait). An evasion becomes `(:Utterance)-[:RAISED]->(:Flag)-[:ABOUT_QUESTION]->(:Utterance)` and the follow-up question is whispered, e.g. *"Not answered: monthly churn. Ask: can you share your exact monthly churn percentage?"*
- **Hedging:** qualifiers on checkable facts ("roughly 200", "almost $20M") are tagged on the claim and in the memo.
- **Shifting answers:** a number that changes for the same fact, earlier in the call or on a previous call with the same founder or company, is linked `(:Claim)-[:SHIFTED_FROM]->(:Claim)` and whispered: *"earlier 200, now 150, data shows 84"*.

The memo gets an **Evasions and shifting answers** section quoting both versions.

**People data beyond jobs:** founders are enriched at prep with skills, GitHub (repos, stars, languages, orgs), follower counts and, where your Crustdata key allows it, the profile **authenticity** verdict, which is shown as a badge and listed as a red flag when not genuine. Click any person and choose **Overlap with founder** for shared employers with overlapping tenure, GitHub orgs, skills and repo languages.

**Explore the graph:** click a company to enrich it or **find people** (current/former, by title) via Crustdata people search; click a person to enrich their profile or find colleagues at each past employer; click an investor to enrich it. Results add to the graph with one click.

**Grow the dossier mid-call:** when the founder names a company or person that isn't in the dossier (a competitor, past employer, advisor), a **➕ chip** suggests adding it, or you can type a domain, company name or profile URL into **Add**. It's enriched from Crustdata, linked to the call (`(:Session)-[:ADDED]->`), and becomes checkable: "Apollo only has 100 people" is checked against Apollo's own headcount, and claims made before you added it are re-checked automatically.

**End call → memo** writes the cited memo and saves it to the session.

Reset between demos: `python -m scripts.reset`
