"""Build supervised (features, next-day target) data from the daily series."""
import pandas as pd

LAGS = (0, 1, 2, 3, 7)  # lag_0 = today's value


def make_dataset(daily, horizon=1):
    """Features known at end of day t -> target: PM2.5 on day t+horizon.

    Only rows with a valid today and a valid target are kept.
    """
    s = daily["pm25"]
    X = pd.DataFrame(index=daily.index)
    for k in LAGS:
        X[f"lag_{k}"] = s.shift(k)
    X["roll_mean_3"] = s.rolling(3, min_periods=2).mean()
    X["roll_mean_7"] = s.rolling(7, min_periods=4).mean()
    X["roll_std_7"] = s.rolling(7, min_periods=4).std()
    X["dow"] = (X.index + pd.Timedelta(days=horizon)).dayofweek  # weekday of the target day
    y = s.shift(-horizon).rename("target")
    keep = X["lag_0"].notna() & y.notna()
    return X[keep], y[keep]


def chronological_split(X, y, train=0.70, val=0.15):
    """Time-ordered split. No shuffling, ever."""
    n = len(X)
    i, j = int(n * train), int(n * (train + val))
    return (X.iloc[:i], y.iloc[:i]), (X.iloc[i:j], y.iloc[i:j]), (X.iloc[j:], y.iloc[j:])