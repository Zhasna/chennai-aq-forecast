"""Count days with data per candidate PM2.5 sensor (OpenAQ v3 daily aggregates)."""
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()
BASE = "https://api.openaq.org/v3"
API_KEY = os.getenv("OPENAQ_API_KEY")

CANDIDATES = {
    "Chennai (AirNow)":  (5077812,  "2022-10-03"),
    "Velachery":         (12235531, "2025-02-18"),
    "Manali":            (12235653, "2025-02-18"),
    "Manali Village":    (12235796, "2025-02-18"),
    "Royapuram":         (12236299, "2025-02-18"),
    "Kodungaiyur":       (12236308, "2025-02-18"),
    "Arumbakkam":        (12236274, "2025-02-18"),
    "Perungudi":         (12236290, "2025-02-18"),
}
END = "2026-10-15"


def fetch_days(sensor_id, start):
    rows, page = [], 1
    while True:
        r = requests.get(
            f"{BASE}/sensors/{sensor_id}/days",
            headers={"X-API-Key": API_KEY},
            params={"datetime_from": start, "datetime_to": END,
                    "limit": 1000, "page": page},
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(int(r.headers.get("x-ratelimit-reset", 10)))
            continue
        r.raise_for_status()
        batch = r.json()["results"]
        rows += batch
        time.sleep(1)
        if len(batch) < 1000:
            return rows
        page += 1


def main():
    if not API_KEY:
        sys.exit("OPENAQ_API_KEY not found in .env")
    print(f"{'station':<18}{'days_with_data':>15}{'span_days':>11}{'coverage%':>11}  longest_gap_days")
    for name, (sid, start) in CANDIDATES.items():
        rows = fetch_days(sid, start)
        days = sorted({x["period"]["datetimeFrom"]["utc"][:10] for x in rows})
        if not days:
            print(f"{name:<18}{'0':>15}")
            continue
        import pandas as pd
        idx = pd.to_datetime(days)
        span = (idx[-1] - idx[0]).days + 1
        gaps = idx.to_series().diff().dt.days.dropna()
        longest = int(gaps.max() - 1) if len(gaps) else 0
        print(f"{name:<18}{len(days):>15}{span:>11}{100*len(days)/span:>10.0f}%  {longest}")


if __name__ == "__main__":
    main()