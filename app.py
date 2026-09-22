import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


st.title("🏠 California House Price Prediction App")
st.write("this app predict the house price on the basis of user input")

@st.cache_resource
def load_and_train_model():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X

model, X = load_and_train_model()

st.header("fill details:")

med_inc = st.number_input("Median Income :", value=3.5)
house_age = st.number_input("House Age :", value=20.0)
ave_rooms = st.number_input("Average Rooms :", value=5.0)
ave_bedrms = st.number_input("Average Bedrooms:", value=1.0)
population = st.number_input("Population :", value=1000.0)
ave_occup = st.number_input("Average Occupancy :", value=3.0)
latitude = st.number_input("Latitude:", value=34.0)
longitude = st.number_input("Longitude:", value=-118.0)

# Prediction Button
if st.button("Predict Price"):
    
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    
    prediction = model.predict(input_data)
    final_price = prediction[0] * 100000  
    
    st.success(f"Estimated House Price: ${final_price:,.2f}")