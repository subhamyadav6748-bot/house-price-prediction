"""Synthetic housing dataset generation and loading.

No real housing dataset is bundled, so a realistic synthetic one is generated.
Prices follow a multiplicative structure (price per sq. ft depends on the city,
neighbourhood quality, age and distance from the centre) plus amenity premiums
and noise, so non-linear models have something real to learn. A few values are
set to missing so the imputation step is exercised.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CITIES, DATA_PATH, FURNISHING, RANDOM_STATE

# Approximate base price per sq. ft (INR) used to generate the data.
CITY_BASE_PPSF = {
    "Delhi": 9500,
    "Mumbai": 19000,
    "Bengaluru": 8500,
    "Pune": 7200,
    "Hyderabad": 6800,
}
FURNISHING_PREMIUM = {"Unfurnished": 0.0, "Semi-furnished": 0.04, "Furnished": 0.09}


def generate_dataset(n: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic dataset of `n` houses."""
    rng = np.random.default_rng(seed)

    city = rng.choice(CITIES, n, p=[0.25, 0.20, 0.25, 0.15, 0.15])
    area = np.clip(rng.lognormal(np.log(1150), 0.38, n), 400, 4500).round()
    bedrooms = np.clip(np.round(area / 480 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 5)
    age = rng.integers(0, 31, n).astype(float)
    location = np.clip(rng.normal(6, 1.8, n), 1, 10).round(1)
    distance = np.clip(rng.gamma(2.5, 3.5, n), 0.5, 35).round(1)
    parking = rng.choice([0, 1, 2], n, p=[0.25, 0.55, 0.20])
    furnishing = rng.choice(FURNISHING, n, p=[0.35, 0.40, 0.25])

    base = np.array([CITY_BASE_PPSF[c] for c in city], dtype=float)
    ppsf = (
        base
        * (1 + 0.07 * (location - 6))
        * np.exp(-0.011 * age)
        * np.exp(-0.012 * distance)
        * (1 + np.array([FURNISHING_PREMIUM[f] for f in furnishing]))
    )
    value = ppsf * area + parking * 2.5e5 + (bathrooms - 1) * 1.5e5
    value *= rng.lognormal(0, 0.07, n)  # market noise

    df = pd.DataFrame(
        {
            "City": city,
            "Area_sqft": area.astype(int),
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age_years": age,
            "Location_Score": location,
            "Distance_to_Center_km": distance,
            "Parking_Spaces": parking,
            "Furnishing": furnishing,
            "Price_Lakhs": (value / 1e5).round(2),
        }
    )

    # Introduce a small amount of missing data.
    for col, frac in [("Age_years", 0.02), ("Location_Score", 0.02), ("Distance_to_Center_km", 0.015)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    return df


def load_data(path: Path = DATA_PATH, n: int = 2000, regenerate: bool = False) -> pd.DataFrame:
    """Load the CSV at `path`, generating and saving it first if needed."""
    path = Path(path)
    if regenerate or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n=n)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)
