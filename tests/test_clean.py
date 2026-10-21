import pandas as pd

from aqforecast.clean import clean_hourly, to_daily


def make_raw(values, observed=4):
    ts = pd.date_range("2025-01-01", periods=len(values), freq="h",
                       tz="Asia/Kolkata").tz_convert("UTC")
    return pd.DataFrame({
        "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pm25": values,
        "expected": 4,
        "observed": observed,
    })


def test_zeros_become_missing():
    out, stats = clean_hourly(make_raw([20.0, 0.0, 22.0]))
    assert out.pm25.isna().tolist() == [False, True, False]
    assert stats["zeros"] == 1


def test_low_coverage_hour_masked():
    out, _ = clean_hourly(make_raw([20.0, 21.0, 22.0], observed=[4, 2, 4]))
    assert out.pm25.isna().tolist() == [False, True, False]


def test_stuck_run_masked():
    out, stats = clean_hourly(make_raw([29.3] * 6 + [10.0, 12.0]))
    assert out.pm25.iloc[:6].isna().all()
    assert out.pm25.iloc[6:].notna().all()
    assert stats["stuck"] == 6


def test_short_repeat_kept():
    out, _ = clean_hourly(make_raw([29.3] * 5 + [10.0]))
    assert out.pm25.notna().all()


def test_real_spike_kept():
    out, _ = clean_hourly(make_raw([20.0, 843.0, 25.0]))
    assert out.pm25.iloc[1] == 843.0


def test_daily_requires_enough_hours():
    full, _ = clean_hourly(make_raw([10.0 + i for i in range(24)]))
    assert to_daily(full).pm25.notna().all()

    sparse, _ = clean_hourly(make_raw([10.0 + i for i in range(10)]))
    assert to_daily(sparse).pm25.isna().all()