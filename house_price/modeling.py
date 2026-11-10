"""Preprocessing pipeline, candidate models and evaluation metrics."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BASELINE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones."""
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, list(numeric)), ("cat", categorical_pipe, list(categorical))]
    )


def make_estimator(model) -> TransformedTargetRegressor:
    """Full pipeline: preprocessing + model, trained on log(price).

    House prices are right-skewed and errors scale with price, so the model is
    fitted on log1p(price) and predictions are converted back to price.
    """
    pipe = Pipeline([("pre", build_preprocessor()), ("model", model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def make_baseline() -> Pipeline:
    """The original project's approach: plain linear regression on 3 features."""
    pre = ColumnTransformer(
        [("sel", SimpleImputer(strategy="median"), list(BASELINE_FEATURES))]
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def candidate_models(seed: int = RANDOM_STATE) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }


def tuning_grids() -> dict:
    """Small hyperparameter grids for the two tree-based models."""
    return {
        "Gradient Boosting": {
            "regressor__model__n_estimators": [200, 400],
            "regressor__model__learning_rate": [0.05, 0.1],
            "regressor__model__max_depth": [2, 3, 4],
        },
        "Random Forest": {
            "regressor__model__n_estimators": [200, 400],
            "regressor__model__max_depth": [None, 12],
            "regressor__model__min_samples_leaf": [1, 3],
        },
    }


def evaluate(y_true, y_pred) -> dict:
    """MAE, RMSE (in lakhs), R^2 and MAPE (%)."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }
