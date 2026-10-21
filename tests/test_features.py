import pandas as pd

from aqforecast.features import make_dataset

NAN = float("nan")


def make_daily(vals):
    idx = pd.date_range("2025-01-01", periods=len(vals), freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({"pm25": vals}, index=idx)


def test_target_is_next_day():
    X, y = make_dataset(make_daily([10.0, 20.0, 30.0, 40.0]))
    assert X["lag_0"].tolist() == [10.0, 20.0, 30.0]
    assert y.tolist() == [20.0, 30.0, 40.0]


def test_rows_missing_today_or_target_dropped():
    X, y = make_dataset(make_daily([10.0, NAN, 30.0, 40.0]))
    assert X["lag_0"].tolist() == [30.0]
    assert y.tolist() == [40.0]


def test_features_never_use_the_future():
    base = make_daily([float(i) for i in range(1, 15)])
    changed = base.copy()
    changed.iloc[-1, 0] = 999.0  # change only the last day
    X1, _ = make_dataset(base)
    X2, _ = make_dataset(changed)
    pd.testing.assert_frame_equal(X1, X2)  # features of every kept row are unchanged