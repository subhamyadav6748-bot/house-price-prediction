"""Market-calibrated house price model for 9 Indian cities (Delhi, Mumbai, Bengaluru, Hyderabad, Pune,
Chennai, Kolkata, Gurugram, Noida).

WHAT THIS IS (read before using on a resume):
There is no real transaction dataset bundled here. The training data is SIMULATED: every synthetic
listing's price is built from published 2026 locality ASKING rates (RATES below) plus adjustments for
size, age, furnishing and parking, plus random noise. This keeps predictions in a realistic range for
each locality, but it does NOT prove real-world accuracy. Update RATES with your own source, or replace
simulate_listings() with a real dataset, when you have one.

Run:  python -m house_price.market     (trains, saves the model, prints sanity checks)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "market_model.joblib"

# Approximate average ASKING prices in Rs per sq ft, compiled from property-portal / market-report pages
# published in 2026 (Mar-Oct). Asking prices are usually higher than final sale prices, and portals disagree
# with each other by 20-60% for the same locality, so mid-range values are used. Edit freely.
RATES_AS_OF = "2026"
RATES = {
    "Delhi": {
        "R.K. Puram": 50400, "Golf Links": 48350, "Greater Kailash I": 33000, "Green Park": 32000,
        "South Delhi (average)": 25400, "New Friends Colony": 24500, "Kirti Nagar": 22750, "Janakpuri": 20200,
        "West Delhi (average)": 18350, "Punjabi Bagh": 18300, "North Delhi (average)": 12150, "Inderpuri": 11350,
        "Dilshad Garden": 10500, "Kapashera": 10200, "Dwarka Sector 21": 8700, "Manglapuri": 8250,
        "Dwarka Sector 26": 7900, "Dabri": 7800, "Mahavir Enclave": 5900,
    },
    "Mumbai": {
        "Worli / Malabar Hill": 105000, "Bandra West": 65650, "Juhu": 58650, "Dadar": 44500, "Andheri West": 41550,
        "Powai": 40150, "Goregaon West": 33000, "Chembur": 32650, "Ghatkopar": 31800, "Vashi (Navi Mumbai)": 31200,
        "Borivali": 29000, "Thane (Majiwada)": 20450, "Mira Road": 15750, "Kalamboli (Navi Mumbai)": 13150,
    },
    "Bengaluru": {
        "Sadashivanagar": 32000, "Koramangala": 16650, "Jayanagar": 15500, "Indiranagar": 15000, "Hebbal": 13000,
        "Whitefield": 11500, "HSR Layout": 11000, "Sarjapur Road": 11000, "JP Nagar": 10550, "Yelahanka": 10350,
        "Bannerghatta Road": 10000, "KR Puram": 9800, "Marathahalli": 8000, "Electronic City": 7200,
    },
    "Hyderabad": {
        "Jubilee Hills": 14100, "Madhapur": 13900, "Hitech City": 13000, "Banjara Hills": 12500,
        "Nanakramguda": 11750, "Gachibowli": 11700, "Kokapet": 11000, "KPHB": 11000, "Kondapur": 10150,
        "Miyapur": 8500, "Kukatpally": 7650, "Kompally": 6950, "Bachupally": 6600, "Boduppal": 5400,
    },
    "Pune": {
        "Boat Club Road": 22000, "Koregaon Park": 18400, "Kalyani Nagar": 16300, "Kothrud": 14400,
        "Magarpatta": 13700, "Viman Nagar": 12700, "Aundh": 12500, "Baner": 11700, "Hadapsar": 11400,
        "Kharadi": 11150, "Wakad": 9350, "Hinjewadi": 8200, "Pimpri Chinchwad": 6650, "Wagholi": 5700,
    },
    "Chennai": {
        "Besant Nagar": 20900, "T. Nagar": 19950, "Adyar": 17250, "Anna Nagar": 16150, "Thiruvanmiyur": 14500,
        "Vadapalani": 11600, "Thoraipakkam": 10050, "Velachery": 10000, "Adambakkam": 10000, "Sholinganallur": 9000,
        "Porur": 8500, "Tambaram": 6900, "Ambattur": 6500, "Avadi": 6000, "Urapakkam": 5450,
    },
    "Kolkata": {
        "Park Street": 17500, "Alipore": 12950, "Salt Lake City": 12400, "New Alipore": 12300, "Ballygunge": 12000,
        "Beleghata": 10550, "Salt Lake Sector V": 10200, "Tangra": 9900, "New Town": 8400, "Lake Town": 7300,
        "Dhakuria": 6750, "Rajarhat": 6500, "EM Bypass": 5800, "Joka": 5650, "Behala": 4400, "Kestopur": 4200,
    },
    "Gurugram": {
        "Golf Course Road": 27000, "Golf Course Extension Road": 18900, "Southern Peripheral Road": 16250,
        "Sohna Road": 15550, "Dwarka Expressway": 14700, "New Gurugram (Sec 82-95)": 10950,
    },
    "Noida": {
        "Sector 50": 14100, "Sector 150": 12500, "Central Noida": 12300, "Noida Expressway": 12100,
        "Sector 137": 10100, "Sector 76": 10450, "Greater Noida West": 8650, "Sector 151": 7900,
        "Noida Extension": 7850,
    },
}
# Flat lookup (locality names are unique across cities)
LOCALITY_RATE = {loc: r for city in RATES.values() for loc, r in city.items()}
LOCALITY_CITY = {loc: c for c, d in RATES.items() for loc in d}
FURNISHING = ["Unfurnished", "Semi-furnished", "Furnished"]
_FURN_FACTOR = {"Unfurnished": 1.00, "Semi-furnished": 1.04, "Furnished": 1.08}
FEATURES = ["locality", "area", "bedrooms", "bathrooms", "parking", "furnishing", "age"]


def _price_factor(area, bedrooms, bathrooms, parking, furnishing, age):
    """Multiplier applied to the locality rate (vectorised)."""
    size = (np.asarray(area) / 1000.0) ** -0.08                 # small flats cost more per sq ft
    park = 1 + 0.025 * np.asarray(parking)
    furn = np.array([_FURN_FACTOR[f] for f in np.atleast_1d(furnishing)])
    old = np.maximum(0.75, 1 - 0.007 * np.asarray(age))         # -0.7% per year, floor at -25%
    bath = 1 + 0.015 * (np.asarray(bathrooms) - np.asarray(bedrooms))
    return size * park * furn * old * bath


def simulate_listings(n=60000, seed=42):
    rng = np.random.default_rng(seed)
    locality = rng.choice(list(LOCALITY_RATE), n)
    area = np.clip(rng.lognormal(np.log(1100), 0.45, n), 350, 5000).round(-1)
    bedrooms = np.clip(np.round(area / 450 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 6)
    parking = rng.choice([0, 1, 2], n, p=[0.25, 0.5, 0.25])
    furnishing = rng.choice(FURNISHING, n, p=[0.35, 0.40, 0.25])
    age = rng.integers(0, 36, n)
    base = np.array([LOCALITY_RATE[x] for x in locality])
    price = area * base * _price_factor(area, bedrooms, bathrooms, parking, furnishing, age) \
        * rng.lognormal(0, 0.12, n)                              # +-12% listing-to-listing noise
    return pd.DataFrame({"locality": locality, "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                         "parking": parking, "furnishing": furnishing, "age": age, "price": price})


def build_pipeline():
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["locality", "furnishing"])],
        remainder="passthrough")
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42))])


def train(save=True):
    df = simulate_listings()
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], np.log(df["price"]), test_size=0.2, random_state=42)
    pipe = build_pipeline().fit(X_tr, y_tr)
    r2 = r2_score(y_te, pipe.predict(X_te))                      # R2 on *simulated* data only
    if save:
        try:
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(pipe, MODEL_PATH)
        except OSError:
            pass                                                 # read-only filesystem: just keep in memory
    return pipe, r2


def get_model():
    """Load the saved model; if missing or incompatible with this scikit-learn version, retrain."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train()[0]


def predict_price(model, city, locality, area, bedrooms, bathrooms, parking, furnishing, age):
    if locality not in RATES.get(city, {}):
        raise ValueError(f"Unknown locality '{locality}' for city '{city}'")
    row = pd.DataFrame([{"locality": locality, "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                         "parking": parking, "furnishing": furnishing, "age": age}])
    price = float(np.exp(model.predict(row[FEATURES])[0]))
    return {"price": price, "per_sqft": price / area, "benchmark_per_sqft": RATES[city][locality]}


def format_price(value):
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} crore"
    return f"₹{value / 1e5:.1f} lakh"


if __name__ == "__main__":
    model, r2 = train()
    n_loc = len(LOCALITY_RATE)
    print(f"Trained on simulated data: {len(RATES)} cities, {n_loc} localities "
          f"(R2 on simulated hold-out = {r2:.3f}; says nothing about real-world accuracy)")
    cases = [("Delhi", "R.K. Puram"), ("Delhi", "Dwarka Sector 26"), ("Mumbai", "Bandra West"), ("Mumbai", "Thane (Majiwada)"),
             ("Bengaluru", "Koramangala"), ("Hyderabad", "Gachibowli"), ("Pune", "Baner"), ("Chennai", "Adyar"),
             ("Kolkata", "New Town"), ("Gurugram", "Golf Course Road"), ("Noida", "Sector 150")]
    for city, loc in cases:
        r = predict_price(model, city, loc, 1200, 3, 2, 1, "Semi-furnished", 5)
        print(f"{city:<10} {loc:<20} 1200 sq ft -> {format_price(r['price']):>14} "
              f"({r['per_sqft']:,.0f}/sq ft vs locality rate {r['benchmark_per_sqft']:,})")
