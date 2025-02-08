import streamlit as st
import joblib
import numpy as np

def load_model(model_path):
    """Load a pre-trained model from a given path."""
    return joblib.load(model_path)

def predict(model, features):
    """Make predictions using the loaded model."""
    return model.predict([features])[0]

# App Layout
st.set_page_config(page_title="Fake Social Media Profile Detection", page_icon=":guardsman:", layout="wide")

# Sidebar Navbar
st.sidebar.title("Navigation")
menu = ["Home", "Algorithms"]
selection = st.sidebar.radio("Go to:", menu)

if selection == "Home":
    st.title("Fake Social Media Profile Detection")
    st.image("https://via.placeholder.com/800x400.png?text=Social+Media+Detection", caption="Identify fake profiles with advanced algorithms.")
    st.write("This application helps detect fake social media profiles using machine learning algorithms. Choose an algorithm from the sidebar to start.")

elif selection == "Algorithms":
    algo_menu = ["SVM", "Logistic Regression", "KNN", "Decision Tree", "Neural Network"]
    algo_choice = st.sidebar.selectbox("Choose Algorithm:", algo_menu)

    # SVM Algorithm
    if algo_choice == "SVM":
        st.header("Support Vector Machine (SVM)")
        model = load_model("svm_model.pkl")
        
        # Input features
        feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01)
        feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01)
        # Add more features as needed

        if st.button("Predict"):
            result = predict(model, [feature1, feature2])
            st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")

    # Logistic Regression Algorithm
    elif algo_choice == "Logistic Regression":
        st.header("Logistic Regression")
        model = load_model("logistic_model.pkl")
        
        # Input features
        feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01)
        feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01)
        # Add more features as needed

        if st.button("Predict"):
            result = predict(model, [feature1, feature2])
            st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")

    # KNN Algorithm
    elif algo_choice == "KNN":
        st.header("K-Nearest Neighbors (KNN)")
        model = load_model("knn_model.pkl")
        
        # Input features
        feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01)
        feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01)
        # Add more features as needed

        if st.button("Predict"):
            result = predict(model, [feature1, feature2])
            st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")

    # Decision Tree Algorithm
    elif algo_choice == "Decision Tree":
        st.header("Decision Tree")
        model = load_model("decision_tree_model.pkl")
        
        # Input features
        feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01)
        feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01)
        # Add more features as needed

        if st.button("Predict"):
            result = predict(model, [feature1, feature2])
            st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")

    # Neural Network Algorithm
    elif algo_choice == "Neural Network":
        st.header("Neural Network")
        model = load_model("neural_network_model.pkl")
        
        # Input features
        feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01)
        feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01)
        # Add more features as needed

        if st.button("Predict"):
            result = predict(model, [feature1, feature2])
            st.success("Prediction: Fake Profile" if result == 1 else "Prediction: Real Profile")
