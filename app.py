import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =========================
# Load Model & Scaler
# =========================

model = load_model("house_price_ann.keras")
scaler = joblib.load("scaler.pkl")

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter housing details to predict house price")

# =========================
# User Inputs
# =========================

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)

total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)

population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)

median_income = st.number_input("Median Income", value=8.3252)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ['INLAND', 'NEAR OCEAN', 'NEAR BAY', '<1H OCEAN', 'ISLAND']
)

# =========================
# Feature Engineering
# =========================

rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# =========================
# One-Hot Encoding
# =========================

INLAND = 0
NEAR_OCEAN = 0
NEAR_BAY = 0
LESS_1H_OCEAN = 0
ISLAND = 0

if ocean_proximity == 'INLAND':
    INLAND = 1

elif ocean_proximity == 'NEAR OCEAN':
    NEAR_OCEAN = 1

elif ocean_proximity == 'NEAR BAY':
    NEAR_BAY = 1

elif ocean_proximity == '<1H OCEAN':
    LESS_1H_OCEAN = 1

elif ocean_proximity == 'ISLAND':
    ISLAND = 1

# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame([[
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    rooms_per_household,
    bedrooms_per_room,
    population_per_household,
    INLAND,
    NEAR_OCEAN,
    NEAR_BAY,
    LESS_1H_OCEAN,
    ISLAND
]], columns=[
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'rooms_per_household',
    'bedrooms_per_room',
    'population_per_household',
    'ocean_proximity_INLAND',
    'ocean_proximity_NEAR OCEAN',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_<1H OCEAN',
    'ocean_proximity_ISLAND'
])

# =========================
# Prediction
# =========================

if st.button("Predict House Price"):

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🏡 Predicted House Price: ${prediction[0][0]:,.2f}")