"""Fetch hourly PM2.5 for candidate sensors, save raw parquet, report quality."""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = "https://api.openaq.org/v3"
API_KEY = os.getenv("OPENAQ_API_KEY")
START, END = "2025-02-18", "2026-10-15"
OUT = Path("data/raw")

SENSORS = {
    "manali_village": 12235796,
    "perungudi": 12236290,
    "velachery": 12235531,
}


def fetch_hours(sensor_id):
    rows, page = [], 1
    while True:
        r = requests.get(
            f"{BASE}/sensors/{sensor_id}/hours",
            headers={"X-API-Key": API_KEY},
            params={"datetime_from": START, "datetime_to": END,
                    "limit": 1000, "page": page},
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(int(r.headers.get("x-ratelimit-reset", 10)))
            continue
        r.raise_for_status()
        batch = r.json()["results"]
        for x in batch:
            cov = x.get("coverage") or {}
            rows.append({
                "ts_utc": x["period"]["datetimeFrom"]["utc"],
                "pm25": x["value"],
                "expected": cov.get("expectedCount"),
                "observed": cov.get("observedCount"),
            })
        print(f"  page {page}: {len(batch)} rows")
        time.sleep(1)
        if len(batch) < 1000:
            return pd.DataFrame(rows)
        page += 1


def report(name, df):
    ts = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert("Asia/Kolkata")
    per_day = ts.dt.date.value_counts().sort_index()
    good = per_day[per_day >= 18] 
    idx = pd.to_datetime(good.index)
    gaps = idx.to_series().diff().dt.days.dropna()
    span = (idx[-1] - idx[0]).days + 1
    print(f"\n{name}: {len(df)} hourly rows")
    print(f"  days with >=18 hours: {len(good)} / {span} ({100*len(good)/span:.0f}%)")
    print(f"  longest gap between good days: {int(gaps.max()-1)} days")
    print(f"  pm25 min/median/max: {df.pm25.min():.1f} / {df.pm25.median():.1f} / {df.pm25.max():.1f}")
    print(f"  negative values: {(df.pm25 < 0).sum()}")


def main():
    if not API_KEY:
        sys.exit("OPENAQ_API_KEY not found in .env")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sid in SENSORS.items():
        print(f"Fetching {name} (sensor {sid})")
        df = fetch_hours(sid)
        df.to_parquet(OUT / f"openaq_pm25_{name}.parquet", index=False)
        report(name, df)


if __name__ == "__main__":
    main()