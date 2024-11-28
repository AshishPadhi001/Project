import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model and data
model_path = "E:\\ML Projects\\Car Price Predictor\\LinearRegressionModel.pkl"
csv_path = "E:\\ML Projects\\Car Price Predictor\\Cleaned_Car_data.xls"

# Load the model
model = pickle.load(open(model_path, 'rb'))

# Load the car data
car_data = pd.read_csv(csv_path)

# Streamlit app title
st.title('Car Price Prediction App')

# Main display for user inputs
st.header('Input Car Details')

# Dropdowns for input fields
company = st.selectbox('Select Car Company', ['Please select a company'] + sorted(car_data['company'].unique()))
car_model = st.selectbox('Select Car Model', ['Please select a model'] + sorted(car_data['name'].unique()))
year = st.selectbox('Select Year of Purchase', ['Please select a year'] + sorted(car_data['year'].unique(), reverse=True))
fuel_type = st.selectbox('Select Fuel Type', ['Please select a fuel type'] + car_data['fuel_type'].unique().tolist())

# Input field for kilometers driven
kms_driven = st.text_input('Enter Kilometers Driven', placeholder='Please enter kilometers driven')

# Predict button
if st.button('Predict Price'):
    # Check if all selections are made and kms_driven is a valid number
    if (company != 'Please select a company' and
        car_model != 'Please select a model' and
        year != 'Please select a year' and
        fuel_type != 'Please select a fuel type' and
        kms_driven.isdigit()):
        
        # Convert kilometers driven to an integer
        kms_driven = int(kms_driven)

        # Data for prediction
        input_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })

        # Making predictions
        prediction = model.predict(input_data)[0]
        
        # Displaying the prediction result
        st.write(f"### Estimated Price: ₹{round(prediction, 2)}")
    else:
        st.error("Please select all input values and ensure kilometers driven is a valid number.")
