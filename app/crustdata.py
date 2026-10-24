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


def _cached(kind: str, body: dict) -> list:
    if not config.CRUSTDATA_API_KEY:
        sample = json.loads(SAMPLE.read_text())
        return sample[kind]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}-{hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = httpx.post(f"{config.CRUSTDATA_BASE_URL}/{kind}/enrich", json=body, timeout=60, headers={
        "authorization": f"Bearer {config.CRUSTDATA_API_KEY}",
        "x-api-version": config.CRUSTDATA_API_VERSION,
        "content-type": "application/json",
    })
    if r.status_code != 200:
        raise CrustdataError(f"POST /{kind}/enrich -> {r.status_code}: {r.text[:300]}")
    data = r.json()
    path.write_text(json.dumps(data, indent=1))
    return data


def _first_match(resp: list, data_key: str) -> dict | None:
    for entry in resp or []:
        matches = entry.get("matches") or []
        if matches:
            return max(matches, key=lambda m: m.get("confidence_score") or 0).get(data_key)
    return None


def enrich_company(domain: str) -> dict | None:
    """Firmographics, headcount timeseries, funding rounds, hiring and founders' profiles."""
    return _first_match(_cached("company", {"domains": [domain], "fields": COMPANY_FIELDS}), "company_data")


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
