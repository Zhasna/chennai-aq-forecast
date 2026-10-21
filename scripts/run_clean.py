from pathlib import Path

import pandas as pd

from aqforecast.clean import clean_hourly, to_daily

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("openaq_pm25_*.parquet")):
        name = f.stem.replace("openaq_pm25_", "")
        hourly, stats = clean_hourly(pd.read_parquet(f))
        daily = to_daily(hourly)
        hourly.to_parquet(OUT / f"{name}_hourly.parquet")
        daily.to_parquet(OUT / f"{name}_daily.parquet")
        print(f"\n{name}: {stats}")
        print(f"  valid days: {daily.pm25.notna().sum()} / {len(daily)}")


if __name__ == "__main__":
    main()