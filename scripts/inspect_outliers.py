import pandas as pd

df = pd.read_parquet("data/raw/openaq_pm25_manali_village.parquet")
df["ts"] = pd.to_datetime(df.ts_utc, utc=True).dt.tz_convert("Asia/Kolkata")
df = df.sort_values("ts").reset_index(drop=True)

print(df.pm25.describe(percentiles=[.5, .9, .99, .999]).round(1))
print(f"\nzeros: {(df.pm25 == 0).sum()} | >300: {(df.pm25 > 300).sum()} | >=999: {(df.pm25 >= 999).sum()}")

print("\nTop 15 readings:")
print(df.nlargest(15, "pm25")[["ts", "pm25", "observed", "expected"]].to_string(index=False))

print("\nZeros per month:")
print(df[df.pm25 == 0].groupby(df.ts.dt.strftime("%Y-%m")).size())

# stuck sensor check: longest run of identical consecutive values
runs = (df.pm25 != df.pm25.shift()).cumsum()
longest = df.groupby(runs).agg(value=("pm25", "first"), hours=("pm25", "size"), start=("ts", "first"))
print("\nLongest runs of identical values:")
print(longest.nlargest(5, "hours").to_string())