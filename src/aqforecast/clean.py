"""Cleaning rules for raw hourly OpenAQ PM2.5 data."""
import pandas as pd

TZ = "Asia/Kolkata"
MIN_COVERAGE = 0.75   
STUCK_HOURS = 6       
MIN_HOURS_PER_DAY = 18
SENSOR_CEILING = 1000.0


def clean_hourly(raw, min_coverage=MIN_COVERAGE, stuck_hours=STUCK_HOURS):
    """Return (hourly DataFrame with column pm25, stats dict).

    Bad hours become NaN. Nothing is interpolated.
    """
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert(TZ)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")

    # full hourly grid so gaps are explicit NaNs (needed for run detection)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(grid)
    df.index.name = "ts"

    v = df["pm25"].astype(float)
    stats = {"rows_raw": len(raw), "valid_before": int(v.notna().sum())}

    low = (df["observed"] / df["expected"] < min_coverage) & v.notna()
    v = v.mask(low)
    stats["low_coverage"] = int(low.sum())

    zero = v == 0
    v = v.mask(zero)
    stats["zeros"] = int(zero.sum())

    capped = v >= SENSOR_CEILING
    v = v.mask(capped)
    stats["capped"] = int(capped.sum())

    run_id = (v != v.shift()).cumsum()
    run_len = v.groupby(run_id).transform("size")
    stuck = (run_len >= stuck_hours) & v.notna()
    v = v.mask(stuck)
    stats["stuck"] = int(stuck.sum())

    stats["valid_after"] = int(v.notna().sum())
    return pd.DataFrame({"pm25": v}), stats


def to_daily(hourly, min_hours=MIN_HOURS_PER_DAY):
    """Daily mean (IST days); days with too few valid hours become NaN."""
    g = hourly["pm25"].resample("D")
    daily = pd.DataFrame({"pm25": g.mean(), "n_hours": g.count()})
    daily.loc[daily["n_hours"] < min_hours, "pm25"] = float("nan")
    return daily