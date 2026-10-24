"""Crustdata API client (https://docs.crustdata.com, API version 2025-11-01).

Responses are cached in data/cache/ so a demo never spends credits twice (the default rate limit
is 15 requests/minute). Without CRUSTDATA_API_KEY the client serves data/sample_dossier.json,
a fictional founder and company in the same response shape.
"""

import hashlib
import json

import httpx

from app import config

CACHE = config.ROOT / "data" / "cache"
SAMPLE = config.ROOT / "data" / "sample_dossier.json"

COMPANY_FIELDS = ["basic_info", "headcount", "funding", "hiring", "people"]
PERSON_FIELDS = ["basic_profile", "experience", "education", "skills"]


class CrustdataError(RuntimeError):
    pass


def _post(path_: str, body: dict):
    """POST to Crustdata with an on-disk cache keyed by endpoint + body."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = path_.strip("/").replace("/", "-")
    path = CACHE / f"{name}-{hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = httpx.post(f"{config.CRUSTDATA_BASE_URL}{path_}", json=body, timeout=60, headers={
        "authorization": f"Bearer {config.CRUSTDATA_API_KEY}",
        "x-api-version": config.CRUSTDATA_API_VERSION,
        "content-type": "application/json",
    })
    if r.status_code != 200:
        raise CrustdataError(f"POST {path_} -> {r.status_code}: {r.text[:300]}")
    data = r.json()
    path.write_text(json.dumps(data, indent=1))
    return data


def _cached(kind: str, body: dict) -> list:
    if not config.CRUSTDATA_API_KEY:
        sample = json.loads(SAMPLE.read_text())
        return sample[kind]
    return _post(f"/{kind}/enrich", body)


def _first_match(resp: list, data_key: str) -> dict | None:
    for entry in resp or []:
        matches = entry.get("matches") or []
        if matches:
            return max(matches, key=lambda m: m.get("confidence_score") or 0).get(data_key)
    return None


def enrich_company(domain: str | None = None, name: str | None = None) -> dict | None:
    """Firmographics, headcount timeseries, funding rounds, hiring and founders' profiles,
    looked up by domain or, for a company mentioned on a call, by name."""
    body = {"domains": [domain]} if domain else {"names": [name]}
    return _first_match(_cached("company", {**body, "fields": COMPANY_FIELDS}), "company_data")


def search_people(domain: str | None = None, company: str | None = None, when: str = "any",
                  title: str | None = None, limit: int = 10) -> list[dict]:
    """People search (0.03 credits/result with basic fields): current and/or former employees of a company,
    optionally by title. when: "current", "past" or "any"."""
    if not config.CRUSTDATA_API_KEY:
        raise CrustdataError("People search needs CRUSTDATA_API_KEY.")
    def employer(kind):
        if domain:
            return {"field": f"experience.employment_details.{kind}.company_website_domain", "type": "=", "value": domain}
        return {"field": f"experience.employment_details.{kind}.company_name", "type": "(.)", "value": company}
    kinds = ["current", "past"] if when == "any" else [when]
    at = employer(kinds[0]) if len(kinds) == 1 else {"op": "or", "conditions": [employer(k) for k in kinds]}
    conditions = [at]
    if title:
        tk = kinds[0] if len(kinds) == 1 else None
        title_conds = [{"field": f"experience.employment_details.{k}.title", "type": "(.)", "value": title}
                       for k in ([tk] if tk else ["current", "past"])]
        conditions.append(title_conds[0] if len(title_conds) == 1 else {"op": "or", "conditions": title_conds})
    body = {"filters": {"op": "and", "conditions": conditions} if len(conditions) > 1 else at,
            "fields": ["basic_profile.name", "basic_profile.headline", "basic_profile.current_title",
                       "social_handles"], "limit": limit}
    out = []
    for p in _post("/person/search", body).get("profiles") or []:
        basic = p.get("basic_profile") or {}
        url = (((p.get("social_handles") or {}).get("professional_network_identifier") or {}).get("profile_url"))
        out.append({"pid": str(p.get("crustdata_person_id")), "name": basic.get("name"),
                    "headline": basic.get("headline") or basic.get("current_title"), "url": url})
    return out


def enrich_people(profile_urls: list[str]) -> list[dict]:
    """Employment history with titles and dates, education and skills, for up to 25 profiles in one call.
    Returns person_data in input order; profiles Crustdata can't match are skipped."""
    resp = _cached("person", {"professional_network_profile_urls": profile_urls[:25], "fields": PERSON_FIELDS})
    by_url = {e.get("matched_on"): e for e in resp or []}
    people = []
    for url in profile_urls[:25]:
        entry = by_url.get(url) or (resp[0] if not config.CRUSTDATA_API_KEY and resp else None)
        match = _first_match([entry], "person_data") if entry else None
        if match and match not in people:
            people.append(match)
    return people
