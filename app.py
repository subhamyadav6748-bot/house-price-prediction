"""Streamlit demo: enter house details and get a price estimate.

Run with:  streamlit run app.py   (train the model first: python -m house_price.train)
"""

import streamlit as st

from house_price.config import CITIES, FURNISHING
from house_price.predict import format_price, load_model, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.caption("Machine-learning price estimate trained on a synthetic housing dataset.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", CITIES, index=2)
    area = st.number_input("Area (sq. ft)", 400, 4500, 1200, step=50)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    parking = st.slider("Parking spaces", 0, 2, 1)
with col2:
    furnishing = st.selectbox("Furnishing", FURNISHING, index=1)
    age = st.slider("Age of house (years)", 0, 30, 5)
    location = st.slider("Location score (1-10)", 1.0, 10.0, 6.0, step=0.1)
    distance = st.slider("Distance to city centre (km)", 0.5, 35.0, 8.0, step=0.5)

if st.button("Predict price", type="primary"):
    price = predict_price(
        {
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age_years": age,
            "Location_Score": location,
            "Distance_to_Center_km": distance,
            "Parking_Spaces": parking,
            "City": city,
            "Furnishing": furnishing,
        },
        model,
    )
    st.metric("Estimated price", format_price(price))
