"""Paths, feature lists and constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = DATA_DIR / "house_data.csv"
MODEL_PATH = MODELS_DIR / "house_price_model.joblib"

RANDOM_STATE = 42
TARGET = "Price_Lakhs"  # price in INR lakhs (1 lakh = 100,000)

NUMERIC_FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Location_Score",
    "Distance_to_Center_km",
    "Parking_Spaces",
]
CATEGORICAL_FEATURES = ["City", "Furnishing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The three features used by the first version of this project.
BASELINE_FEATURES = ["Area_sqft", "Bedrooms", "Location_Score"]

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad"]
FURNISHING = ["Unfurnished", "Semi-furnished", "Furnished"]
