"""First XGBoost model vs persistence on identical rows (chronological split)."""
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from aqforecast.evaluate import mae, rmse
from aqforecast.features import chronological_split, make_dataset

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["manali_village", "perungudi"]

PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,           # shallow: ~260 training rows
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,  # uses the validation set only
    random_state=42,
)


def main():
    for name in STATIONS:
        daily = pd.read_parquet(ROOT / "data" / "processed" / f"{name}_daily.parquet")
        X, y = make_dataset(daily)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y)

        model = XGBRegressor(**PARAMS)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        print(f"\n{name}  (best iteration: {model.best_iteration})")

        for split, Xs, ys in [("val", Xva, yva), ("test", Xte, yte)]:
            pm, pb = model.predict(Xs), Xs["lag_0"]
            m_mae, b_mae = mae(ys, pm), mae(ys, pb)
            print(f"  {split:<4} n={len(Xs)}  "
                  f"model MAE={m_mae:5.2f} RMSE={rmse(ys, pm):5.2f} | "
                  f"persistence MAE={b_mae:5.2f} RMSE={rmse(ys, pb):5.2f} | "
                  f"MAE skill={1 - m_mae / b_mae:+.1%}")

        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("  feature importance:", ", ".join(f"{k}={v:.2f}" for k, v in imp.items()))

        # which test days drive the error?
        t = pd.DataFrame({"actual": yte, "persistence": Xte["lag_0"],
                          "model": model.predict(Xte)})
        t["persist_err"] = (t.actual - t.persistence).abs()
        print("  worst test days for persistence:")
        print(t.nlargest(5, "persist_err").round(1).to_string())


if __name__ == "__main__":
    main()