from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = "posted_rate"
CAT_COLUMNS = ["pickup", "delivery", "equipment", "route"]
NUM_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "month", "day_of_week",
    "day_of_month", "day_of_year", "time_index", "distance_log", "weight_missing",
    "weight_negative", "weight_clean", "weight_per_mile", "latitude_delta", "longitude_delta",
]
FEATURE_COLUMNS = CAT_COLUMNS + NUM_COLUMNS


def add_features(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create features using row inputs and training-derived fallbacks only."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")

    if reference is not None:
        reference = reference.copy()
        reference["date"] = pd.to_datetime(reference["date"], errors="raise")
        mappings = {
            "pickup_lat": result["pickup"].map(reference.groupby("pickup")["pickup_lat"].mean()),
            "pickup_lon": result["pickup"].map(reference.groupby("pickup")["pickup_lon"].mean()),
            "delivery_lat": result["delivery"].map(reference.groupby("delivery")["delivery_lat"].mean()),
            "delivery_lon": result["delivery"].map(reference.groupby("delivery")["delivery_lon"].mean()),
        }
        global_coords = {
            "pickup_lat": reference["pickup_lat"].mean(), "pickup_lon": reference["pickup_lon"].mean(),
            "delivery_lat": reference["delivery_lat"].mean(), "delivery_lon": reference["delivery_lon"].mean(),
        }
        for column, values in mappings.items():
            if column not in result:
                result[column] = values
            else:
                result[column] = result[column].fillna(values)
            result[column] = result[column].fillna(global_coords[column])
        for column in ["market_index", "quote_signal"]:
            fallback = reference[column].median()
            result[column] = result[column].fillna(fallback) if column in result else fallback

    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"]:
        if column not in result:
            result[column] = np.nan

    result["route"] = result["pickup"].astype(str) + " -> " + result["delivery"].astype(str)
    result["month"] = result["date"].dt.month
    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_month"] = result["date"].dt.day
    result["day_of_year"] = result["date"].dt.dayofyear
    result["time_index"] = (result["date"] - pd.Timestamp("2025-01-01")).dt.days
    result["distance_log"] = np.log1p(result["distance"])
    result["weight_missing"] = result["weight"].isna().astype(int)
    result["weight_negative"] = (result["weight"] < 0).fillna(False).astype(int)
    result["weight_clean"] = result["weight"].where(result["weight"] >= 0)
    result["weight_per_mile"] = result["weight_clean"] / result["distance"]
    result["latitude_delta"] = result["delivery_lat"] - result["pickup_lat"]
    result["longitude_delta"] = result["delivery_lon"] - result["pickup_lon"]
    return result


def score(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {"MAE": float(mean_absolute_error(y_true, prediction)), "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction)))}


def train_catboost(train: pd.DataFrame) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=700, depth=8, learning_rate=0.06, loss_function="RMSE",
        random_seed=42, l2_leaf_reg=5.0, verbose=False, allow_writing_files=False,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET], cat_features=CAT_COLUMNS)
    return model


def train_ridge(train: pd.DataFrame) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLUMNS),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLUMNS),
    ])
    model = Pipeline([("features", transformer), ("regressor", Ridge(alpha=10.0))])
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    return model


def make_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp("2025-09-01")
    return train[train["date"] < cutoff].copy(), train[train["date"] >= cutoff].copy()


def run_experiments(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    train_raw = train_raw.copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    dev_raw, holdout_raw = make_split(train_raw)
    dev, holdout = add_features(dev_raw, dev_raw), add_features(holdout_raw, dev_raw)
    rows = [{"model": "Median baseline", **score(holdout[TARGET], np.full(len(holdout), dev[TARGET].median()))}]

    started = time.perf_counter()
    ridge = train_ridge(dev)
    rows.append({"model": "Ridge", **score(holdout[TARGET], ridge.predict(holdout[FEATURE_COLUMNS])), "seconds": round(time.perf_counter() - started, 2)})
    started = time.perf_counter()
    catboost = train_catboost(dev)
    rows.append({"model": "CatBoost", **score(holdout[TARGET], catboost.predict(holdout[FEATURE_COLUMNS])), "seconds": round(time.perf_counter() - started, 2)})
    details = {"split": {
        "development_start": str(dev_raw["date"].min().date()), "development_end": str(dev_raw["date"].max().date()),
        "holdout_start": str(holdout_raw["date"].min().date()), "holdout_end": str(holdout_raw["date"].max().date()),
        "development_rows": len(dev_raw), "holdout_rows": len(holdout_raw),
    }, "catboost_model": {"iterations": 700, "depth": 8, "learning_rate": 0.06, "random_seed": 42}}
    return pd.DataFrame(rows), details


def train_final_and_predict(train_raw: pd.DataFrame, validation_raw: pd.DataFrame, december_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, validation_raw, december_raw = train_raw.copy(), validation_raw.copy(), december_raw.copy()
    for frame in [train_raw, validation_raw, december_raw]:
        frame["date"] = pd.to_datetime(frame["date"])
    train, validation, december = add_features(train_raw, train_raw), add_features(validation_raw, train_raw), add_features(december_raw, train_raw)
    model = train_catboost(train)
    validation_output = validation_raw[["load_id"]].copy()
    validation_output["predicted_rate"] = np.maximum(model.predict(validation[FEATURE_COLUMNS]), 0.01)
    december_output = december_raw.copy()
    december_output["predicted_rate"] = np.maximum(model.predict(december[FEATURE_COLUMNS]), 0.01)
    return validation_output, december_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, and generate Spotter freight-rate outputs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train-test.csv")
    validation_raw = pd.read_csv(DATA_DIR / "validation.csv")
    validation_template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")
    december_raw = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")
    results, details = run_experiments(train_raw)
    results.to_csv(args.output_dir / "model_results.csv", index=False)
    (args.output_dir / "run_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    validation_output, december_output = train_final_and_predict(train_raw, validation_raw, december_raw)
    if not validation_template["load_id"].equals(validation_output["load_id"]):
        raise ValueError("validation prediction IDs do not match the supplied template")
    validation_template["predicted_rate"] = validation_output["predicted_rate"].to_numpy()
    validation_template.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    december_output.to_csv(DATA_DIR / "december-chart-inputs.csv", index=False)
    print(f"Wrote {len(validation_output):,} validation predictions.")
    print(f"Wrote {len(december_output):,} December predictions.")


if __name__ == "__main__":
    main()