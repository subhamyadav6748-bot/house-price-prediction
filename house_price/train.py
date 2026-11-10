"""Train, compare, tune and evaluate models.

Usage:
    python -m house_price.train              # full run with hyperparameter tuning
    python -m house_price.train --no-tune    # faster, skips grid search
    python -m house_price.train --regenerate # rebuild the synthetic dataset first
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split

from . import plots
from .config import (
    DATA_PATH,
    FEATURES,
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from .data import load_data
from .modeling import candidate_models, evaluate, make_baseline, make_estimator, tuning_grids

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cv_scores(estimator, X, y, cv) -> dict:
    res = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {
        "cv_RMSE": -res["test_rmse"].mean(),
        "cv_RMSE_std": res["test_rmse"].std(),
        "cv_R2": res["test_r2"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-tune", action="store_true", help="skip hyperparameter search")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the dataset")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    # 1. Data
    df = load_data(DATA_PATH, n=args.n_samples, regenerate=args.regenerate)
    print(f"Loaded {len(df):,} rows, {df.isna().any(axis=1).sum()} rows with missing values")
    plots.plot_eda(df, FIGURES_DIR / "eda.png")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Baseline (the original 3-feature linear regression) + candidate models
    rows, fitted = [], {}

    baseline = make_baseline()
    row = {"model": "Baseline (3-feature linear)", **cv_scores(baseline, X_train, y_train, cv)}
    baseline.fit(X_train, y_train)
    pred = baseline.predict(X_test)
    row.update({f"test_{k}": v for k, v in evaluate(y_test, pred).items()})
    rows.append(row)
    fitted["Baseline (3-feature linear)"] = baseline

    for name, model in candidate_models().items():
        est = make_estimator(model)
        row = {"model": name, **cv_scores(est, X_train, y_train, cv)}
        est.fit(X_train, y_train)
        row.update({f"test_{k}": v for k, v in evaluate(y_test, est.predict(X_test)).items()})
        rows.append(row)
        fitted[name] = est
        print(f"  {name:<20} CV RMSE {row['cv_RMSE']:8.2f}   CV R2 {row['cv_R2']:.4f}")

    # 3. Hyperparameter tuning for the tree-based models
    best_params = {}
    if not args.no_tune:
        models = candidate_models()
        for name, grid in tuning_grids().items():
            if name == "Random Forest":
                models[name].set_params(n_jobs=1)  # parallelism handled by the grid search
            search = GridSearchCV(
                make_estimator(models[name]),
                grid,
                scoring=SCORING,
                refit="rmse",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            i = search.best_index_
            tuned = f"{name} (tuned)"
            row = {
                "model": tuned,
                "cv_RMSE": -search.cv_results_["mean_test_rmse"][i],
                "cv_RMSE_std": search.cv_results_["std_test_rmse"][i],
                "cv_R2": search.cv_results_["mean_test_r2"][i],
            }
            row.update({f"test_{k}": v for k, v in evaluate(y_test, search.predict(X_test)).items()})
            rows.append(row)
            fitted[tuned] = search.best_estimator_
            best_params[tuned] = {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()}
            print(f"  {tuned:<20} CV RMSE {row['cv_RMSE']:8.2f}   CV R2 {row['cv_R2']:.4f}   {best_params[tuned]}")

    table = pd.DataFrame(rows)

    # 4. Pick the best model using cross-validation only (the test set stays untouched)
    candidates = table[~table["model"].str.startswith("Baseline")]
    best_name = candidates.sort_values("cv_RMSE").iloc[0]["model"]
    best = fitted[best_name]
    y_pred = best.predict(X_test)
    best_row = table[table["model"] == best_name].iloc[0]
    base_row = table[table["model"].str.startswith("Baseline")].iloc[0]
    rmse_gain = (1 - best_row["test_RMSE"] / base_row["test_RMSE"]) * 100
    print(f"\nBest model: {best_name}")
    print(f"  Test RMSE {best_row['test_RMSE']:.2f} lakhs | MAE {best_row['test_MAE']:.2f} | "
          f"R2 {best_row['test_R2']:.4f} | MAPE {best_row['test_MAPE_pct']:.2f}%")
    print(f"  RMSE reduced by {rmse_gain:.1f}% vs the 3-feature baseline")

    # 5. Figures
    plots.plot_model_comparison(table, FIGURES_DIR / "model_comparison.png")
    plots.plot_actual_vs_predicted(y_test, y_pred, FIGURES_DIR / "actual_vs_predicted.png",
                                   title=f"Actual vs predicted ({best_name})")
    plots.plot_residuals(y_test, y_pred, FIGURES_DIR / "residuals.png")
    imp = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                                 scoring="r2", n_jobs=-1)
    plots.plot_importance(FEATURES, imp.importances_mean, imp.importances_std,
                          FIGURES_DIR / "feature_importance.png")

    # 6. Save artefacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH, compress=3)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    metrics = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_model": best_name,
        "best_params": best_params.get(best_name, {}),
        "test_metrics": {k: round(float(best_row[f"test_{k}"]), 4) for k in ["MAE", "RMSE", "R2", "MAPE_pct"]},
        "baseline_test_metrics": {k: round(float(base_row[f"test_{k}"]), 4) for k in ["MAE", "RMSE", "R2", "MAPE_pct"]},
        "rmse_reduction_vs_baseline_pct": round(float(rmse_gain), 1),
        "feature_importance": {
            f: round(float(m), 4) for f, m in sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])
        },
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}, "
          f"metrics and figures to reports/")


if __name__ == "__main__":
    main()
