"""Discover OpenAQ locations around Chennai and report sensor history length.

Uses OpenAQ API v3. Needs OPENAQ_API_KEY in .env.
"""
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.openaq.org/v3"
API_KEY = os.getenv("OPENAQ_API_KEY")
CHENNAI = (13.0827, 80.2707)  # lat, lon
RADIUS_M = 25_000             # API maximum


def get(path, params=None):
    for attempt in range(3):
        r = requests.get(
            f"{BASE}{path}",
            headers={"X-API-Key": API_KEY},
            params=params,
            timeout=30,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("x-ratelimit-reset", 10))
            print(f"Rate limited, sleeping {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(1) 
        return r.json()
    raise RuntimeError(f"Failed after retries: {path}")


def utc(obj):
    """datetimeFirst/Last come back as {'utc': ..., 'local': ...} or None."""
    return obj["utc"][:10] if obj and obj.get("utc") else "n/a"


def main():
    if not API_KEY:
        sys.exit("OPENAQ_API_KEY not found. Check your .env file.")

    data = get(
        "/locations",
        {
            "coordinates": f"{CHENNAI[0]},{CHENNAI[1]}",
            "radius": RADIUS_M,
            "limit": 100,
        },
    )
    locations = data["results"]
    print(f"Found {len(locations)} locations within {RADIUS_M/1000:.0f} km\n")

    for loc in locations:
        provider = (loc.get("provider") or {}).get("name", "?")
        print(f"[{loc['id']}] {loc['name']}  (provider: {provider})")
        print(f"    location-level: {utc(loc.get('datetimeFirst'))} -> {utc(loc.get('datetimeLast'))}")

        sensors = get(f"/locations/{loc['id']}/sensors")["results"]
        for s in sensors:
            p = s["parameter"]
            print(
                f"    sensor {s['id']:>8}  {p['name']:<6} ({p['units']})  "
                f"{utc(s.get('datetimeFirst'))} -> {utc(s.get('datetimeLast'))}"
            )
        print()


if __name__ == "__main__":
    main()