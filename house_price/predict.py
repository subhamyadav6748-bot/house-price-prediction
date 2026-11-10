"""Predict the price of a single house.

Example:
    python -m house_price.predict --area 1500 --bedrooms 3 --bathrooms 2 \
        --age 5 --location-score 7.5 --distance 6 --parking 1 \
        --city Bengaluru --furnishing Semi-furnished
"""

import argparse

import joblib
import pandas as pd

from .config import CITIES, FEATURES, FURNISHING, MODEL_PATH


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m house_price.train` first."
        )
    return joblib.load(path)


def predict_price(features: dict, model=None) -> float:
    """Return the predicted price in INR lakhs for one house."""
    model = model or load_model()
    row = pd.DataFrame([features])[FEATURES]
    return float(model.predict(row)[0])


def format_price(lakhs: float) -> str:
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} crore"
    return f"₹{lakhs:.1f} lakh"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--area", type=float, required=True, help="area in sq. ft")
    p.add_argument("--bedrooms", type=int, required=True)
    p.add_argument("--bathrooms", type=int, default=2)
    p.add_argument("--age", type=float, default=5, help="age of the house in years")
    p.add_argument("--location-score", type=float, default=6.0, help="neighbourhood quality, 1-10")
    p.add_argument("--distance", type=float, default=8.0, help="distance to city centre in km")
    p.add_argument("--parking", type=int, default=1)
    p.add_argument("--city", choices=CITIES, default="Bengaluru")
    p.add_argument("--furnishing", choices=FURNISHING, default="Semi-furnished")
    a = p.parse_args()

    price = predict_price(
        {
            "Area_sqft": a.area,
            "Bedrooms": a.bedrooms,
            "Bathrooms": a.bathrooms,
            "Age_years": a.age,
            "Location_Score": a.location_score,
            "Distance_to_Center_km": a.distance,
            "Parking_Spaces": a.parking,
            "City": a.city,
            "Furnishing": a.furnishing,
        }
    )
    print(f"Estimated price: {format_price(price)}  ({price:,.1f} lakhs)")


if __name__ == "__main__":
    main()
