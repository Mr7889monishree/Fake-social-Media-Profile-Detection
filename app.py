import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the pre-trained SVM model (make sure to replace this with your own saved model if required)
import joblib

# Assuming you've already trained the model, you can save it as 'svm_model.pkl'
# If you don't have a trained model, use the code to train one and save it using joblib:
# model = SVC()
# model.fit(X_train, y_train)
# joblib.dump(model, 'svm_model.pkl')

# Load the model
svm_model = joblib.load('model/svm_model.pkl')

# Title of the web app
st.title('Fake Profile Detection')

# Function to take input from the user
def get_input():
    input_data = {}
    
    input_data['feature_1'] = st.number_input('Feature 1', min_value=0, max_value=1000, value=50)
    input_data['feature_2'] = st.number_input('Feature 2', min_value=0, max_value=1000, value=50)
    input_data['feature_3'] = st.number_input('Feature 3', min_value=0, max_value=1000, value=50)
    input_data['feature_4'] = st.number_input('Feature 4', min_value=0, max_value=1000, value=50)
    input_data['feature_5'] = st.number_input('Feature 5', min_value=0, max_value=1000, value=50)
    input_data['feature_6'] = st.number_input('Feature 6', min_value=0, max_value=1000, value=50)
    input_data['feature_7'] = st.number_input('Feature 7', min_value=0, max_value=1000, value=50)
    input_data['feature_8'] = st.number_input('Feature 8', min_value=0, max_value=1000, value=50)
    input_data['feature_9'] = st.number_input('Feature 9', min_value=0, max_value=1000, value=50)
    input_data['feature_10'] = st.number_input('Feature 10', min_value=0, max_value=1000000, value=50)
    input_data['feature_11'] = st.number_input('Feature 11', min_value=0, max_value=1000, value=50)

    
    return input_data

# Predict function to use the model for prediction
def predict(input_data):
    # Prepare the data for prediction (standardize the data)
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaler = StandardScaler()
    standardized_input = scaler.fit_transform(input_array)
    
    prediction = svm_model.predict(standardized_input)
    
    if prediction[0] == 0:
        return 'True Profile'
    else:
        return 'Fake Profile'

# UI for taking input
input_data = get_input()

# Prediction button
if st.button('Predict'):
    result = predict(input_data)
    st.write(f'The given profile is a {result}.')
