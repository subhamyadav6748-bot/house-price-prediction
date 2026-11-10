import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from house_price.config import FEATURES, TARGET
from house_price.data import generate_dataset
from house_price.modeling import evaluate, make_baseline, make_estimator
from house_price.predict import format_price, predict_price


@pytest.fixture(scope="module")
def df():
    return generate_dataset(n=400, seed=1)


@pytest.fixture(scope="module")
def model(df):
    est = make_estimator(LinearRegression())
    est.fit(df[FEATURES], df[TARGET])
    return est


def test_dataset_shape_and_target(df):
    assert len(df) == 400
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert df[TARGET].notna().all() and (df[TARGET] > 0).all()


def test_dataset_is_reproducible():
    a = generate_dataset(n=100, seed=7)
    b = generate_dataset(n=100, seed=7)
    assert a.equals(b)


def test_dataset_has_some_missing_values(df):
    assert df[FEATURES].isna().any().any()


def test_pipeline_predicts_positive_prices(df, model):
    pred = model.predict(df[FEATURES])
    assert pred.shape == (len(df),)
    assert (pred > 0).all()


def test_beats_baseline(df):
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], random_state=0)
    base = make_baseline().fit(X_train, y_train)
    full = make_estimator(LinearRegression()).fit(X_train, y_train)
    assert evaluate(y_test, full.predict(X_test))["RMSE"] < evaluate(y_test, base.predict(X_test))["RMSE"]


def test_predict_price_handles_missing_and_unseen_values(model):
    house = {
        "Area_sqft": 1200, "Bedrooms": 3, "Bathrooms": 2, "Age_years": np.nan,
        "Location_Score": 6.5, "Distance_to_Center_km": 8, "Parking_Spaces": 1,
        "City": "Atlantis", "Furnishing": "Semi-furnished",
    }
    assert predict_price(house, model) > 0


def test_bigger_house_costs_more(model):
    house = {
        "Area_sqft": 900, "Bedrooms": 2, "Bathrooms": 2, "Age_years": 5,
        "Location_Score": 6, "Distance_to_Center_km": 8, "Parking_Spaces": 1,
        "City": "Pune", "Furnishing": "Unfurnished",
    }
    small = predict_price(house, model)
    big = predict_price({**house, "Area_sqft": 2500}, model)
    assert big > small


def test_format_price():
    assert format_price(85.0) == "₹85.0 lakh"
    assert format_price(250.0) == "₹2.50 crore"
