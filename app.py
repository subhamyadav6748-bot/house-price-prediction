"""Streamlit demo: house price estimate for 9 Indian cities.  Run with:  streamlit run app.py"""
import streamlit as st

from house_price.market import FURNISHING, RATES, RATES_AS_OF, format_price, get_model, predict_price

st.set_page_config(page_title="India House Price Predictor", page_icon="🏠")
st.title("🏠 India House Price Predictor")
st.caption(f"ML price estimate by city and locality, calibrated to published {RATES_AS_OF} asking-price benchmarks "
           f"({len(RATES)} cities, {sum(len(v) for v in RATES.values())} localities).")
st.warning("Demo model trained on **simulated** data built around published asking prices. "
           "Estimates are illustrative, not a market valuation or financial advice.")


@st.cache_resource
def load():
    return get_model()


model = load()

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", list(RATES), index=0)
    localities = sorted(RATES[city], key=RATES[city].get, reverse=True)
    locality = st.selectbox("Locality", localities, index=len(localities) // 3)
    area = st.number_input("Area (sq. ft)", 300, 5000, 1200, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
with col2:
    bathrooms = st.slider("Bathrooms", 1, 6, 2)
    furnishing = st.selectbox("Furnishing", FURNISHING, index=1)
    parking = st.slider("Parking spaces", 0, 2, 1)
    age = st.slider("Age of house (years)", 0, 35, 5)

if st.button("Predict price", type="primary"):
    result = predict_price(model, city, locality, area, bedrooms, bathrooms, parking, furnishing, age)
    st.metric("Estimated price", format_price(result["price"]))
    c1, c2 = st.columns(2)
    c1.metric("Implied rate", f"₹{result['per_sqft']:,.0f} / sq ft")
    c2.metric(f"{locality} benchmark", f"₹{result['benchmark_per_sqft']:,} / sq ft")
    st.caption("Benchmark = average asking rate for the locality. Asking prices are usually higher than final sale prices, "
               "and listing sites disagree with each other, so treat this as a rough guide.")
