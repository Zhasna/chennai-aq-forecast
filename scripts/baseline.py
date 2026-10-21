"""Persistence baseline ("tomorrow = today") per split. Prints row counts first."""
from pathlib import Path

import pandas as pd

from aqforecast.evaluate import mae, rmse
from aqforecast.features import chronological_split, make_dataset

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["manali_village", "perungudi"]


def main():
    for name in STATIONS:
        daily = pd.read_parquet(ROOT / "data" / "processed" / f"{name}_daily.parquet")
        X, y = make_dataset(daily)
        print(f"\n{name}")
        print(f"  calendar days {len(daily)} | valid days {daily.pm25.notna().sum()} "
              f"| usable (today, tomorrow) pairs {len(X)}")
        print(f"  missing lag_1: {X.lag_1.isna().sum()} | lag_7: {X.lag_7.isna().sum()} "
              f"| roll_mean_7: {X.roll_mean_7.isna().sum()}")
        splits = zip(["train", "val", "test"], chronological_split(X, y))
        for split, (Xs, ys) in splits:
            p = Xs["lag_0"]
            print(f"  {split:<5} n={len(Xs):>3}  {Xs.index.min().date()} -> {Xs.index.max().date()}  "
                  f"mean={ys.mean():5.1f}  persistence MAE={mae(ys, p):5.2f}  RMSE={rmse(ys, p):5.2f}")


if __name__ == "__main__":
    main()