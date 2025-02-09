import streamlit as st
import joblib
import numpy as np
import os

def load_model(model_path):
    """Load a pre-trained model from a given path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict(model, features):
    """Make predictions using the loaded model."""
    try:
        return model.predict([features])[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# App Layout
st.set_page_config(page_title="Fake Social Media Profile Detection", page_icon=":guardsman:", layout="wide")

# Sidebar Navbar
st.sidebar.title("Navigation")
menu = ["Home", "Algorithms"]
selection = st.sidebar.radio("Go to:", menu)

if selection == "Home":
    st.title("Fake Social Media Profile Detection")
    st.image("Images/cyber.webp", caption="Identify fake profiles with advanced algorithms.")
    st.write("This application helps detect fake social media profiles using machine learning algorithms. Choose an algorithm from the sidebar to start.")

elif selection == "Algorithms":
    algo_menu = ["SVM", "Logistic Regression", "KNN", "Decision Tree"]
    algo_choice = st.sidebar.selectbox("Choose Algorithm:", algo_menu)

    # Dynamic Input for Algorithms
    st.header(algo_choice)

    model_paths = {
        "SVM": "models/svm_model.pkl",
        "Logistic Regression": "models/log_reg_model.pkl",
        "KNN": "models/knn_model.pkl",
        "Decision Tree": "models/decision_tree_model.pkl",
    }

    model = load_model(model_paths.get(algo_choice))

    # Input features
    st.subheader("Enter Features")
    features = [st.number_input(f"Feature {i}", min_value=0.0, max_value=100000.0, step=0.01) for i in range(1, 12)]

    if st.button("Predict"):
        if model:
            result = predict(model, features)
            if result is not None:
                st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")
