import pandas as pd
from aqforecast.clean import clean_hourly, to_daily

for name in ["manali_village", "perungudi", "velachery"]:
    raw = pd.read_parquet(f"data/raw/openaq_pm25_{name}.parquet")
    print(f"\n{name}")
    for cov in (0.5, 0.75, 1.0):
        h, s = clean_hourly(raw, min_coverage=cov)
        print(f"  min_coverage={cov}: valid hours {s['valid_after']}, "
              f"valid days {to_daily(h).pm25.notna().sum()}")

p = pd.read_parquet("data/raw/openaq_pm25_perungudi.parquet")
p["ts"] = pd.to_datetime(p.ts_utc, utc=True).dt.tz_convert("Asia/Kolkata")
print("\nPerungudi top readings:")
print(p.nlargest(8, "pm25")[["ts", "pm25", "observed", "expected"]].to_string(index=False))

v = pd.read_parquet("data/raw/openaq_pm25_velachery.parquet")
v["ts"] = pd.to_datetime(v.ts_utc, utc=True).dt.tz_convert("Asia/Kolkata")
v = v.sort_values("ts").reset_index(drop=True)
runs = (v.pm25 != v.pm25.shift()).cumsum()
long = v.groupby(runs).agg(value=("pm25", "first"), hours=("pm25", "size"), start=("ts", "first"))
print("\nVelachery longest identical runs:")
print(long.nlargest(6, "hours").to_string())