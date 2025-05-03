import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

st.set_page_config(page_title="Disease Prediction App", layout="centered")
st.title("\U0001F9E0 AI-Powered Disease Prediction App")
st.write("Predict **Diabetes**, **Heart Disease**, or **Breast Cancer** using Machine Learning")

st.sidebar.subheader("\U0001F6E0️ Debugging Info")
st.sidebar.write("Files in Directory:")
st.sidebar.write(os.listdir())

MODEL_FILES = {
    "diabetes_model": "diabetes_model.pkl",
    "diabetes_scaler": "diabetes_scaler.pkl",
    "heart_model": "heart_model.pkl",
    "heart_scaler": "heart_scaler.pkl",
    "cancer_model": "cancer_model.pkl",
    "cancer_scaler": "cancer_scaler.pkl",
    "cancer_features": "cancer_features.pkl"
}

TRAIN_DATA = {
    "diabetes": None,
    "heart": None,
    "cancer": None
}

def train_and_save_models():
    # Diabetes
    diabetes_df = pd.read_csv("diabetes.csv")
    X_d = diabetes_df.drop("Outcome", axis=1)
    y_d = diabetes_df["Outcome"]
    X_d_train, _, y_d_train, _ = train_test_split(X_d, y_d, test_size=0.2, random_state=0)
    scaler_d = StandardScaler().fit(X_d_train)
    model_d = LogisticRegression().fit(scaler_d.transform(X_d_train), y_d_train)
    joblib.dump(model_d, MODEL_FILES["diabetes_model"])
    joblib.dump(scaler_d, MODEL_FILES["diabetes_scaler"])
    TRAIN_DATA["diabetes"] = (scaler_d.transform(X_d_train), y_d_train)

    # Heart
    heart_df = pd.read_csv("heart.csv")
    X_h = heart_df.drop("target", axis=1)
    y_h = heart_df["target"]
    X_h_train, _, y_h_train, _ = train_test_split(X_h, y_h, test_size=0.2, random_state=0)
    scaler_h = StandardScaler().fit(X_h_train)
    model_h = RandomForestClassifier().fit(scaler_h.transform(X_h_train), y_h_train)
    joblib.dump(model_h, MODEL_FILES["heart_model"])
    joblib.dump(scaler_h, MODEL_FILES["heart_scaler"])
    TRAIN_DATA["heart"] = (scaler_h.transform(X_h_train), y_h_train)

    # Cancer
    cancer = load_breast_cancer()
    X_c = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_c = cancer.target
    X_c_train, _, y_c_train, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=0)
    scaler_c = StandardScaler().fit(X_c_train)
    model_c = LogisticRegression(max_iter=10000).fit(scaler_c.transform(X_c_train), y_c_train)
    joblib.dump(model_c, MODEL_FILES["cancer_model"])
    joblib.dump(scaler_c, MODEL_FILES["cancer_scaler"])
    joblib.dump(cancer.feature_names, MODEL_FILES["cancer_features"])
    TRAIN_DATA["cancer"] = (scaler_c.transform(X_c_train), y_c_train)

if not all(os.path.exists(f) for f in MODEL_FILES.values()):
    st.warning("\U0001F504 Training models for the first time...")
    train_and_save_models()
    st.success("✅ Models trained and saved!")

@st.cache_resource
def load_models():
    diabetes_model = joblib.load(MODEL_FILES["diabetes_model"])
    diabetes_scaler = joblib.load(MODEL_FILES["diabetes_scaler"])
    diabetes_df = pd.read_csv("diabetes.csv")
    X_d = diabetes_df.drop("Outcome", axis=1)
    y_d = diabetes_df["Outcome"]
    X_d_train, _, y_d_train, _ = train_test_split(X_d, y_d, test_size=0.2, random_state=0)
    X_d_train_scaled = diabetes_scaler.transform(X_d_train)

    heart_model = joblib.load(MODEL_FILES["heart_model"])
    heart_scaler = joblib.load(MODEL_FILES["heart_scaler"])
    heart_df = pd.read_csv("heart.csv")
    X_h = heart_df.drop("target", axis=1)
    y_h = heart_df["target"]
    X_h_train, _, y_h_train, _ = train_test_split(X_h, y_h, test_size=0.2, random_state=0)
    X_h_train_scaled = heart_scaler.transform(X_h_train)

    cancer_model = joblib.load(MODEL_FILES["cancer_model"])
    cancer_scaler = joblib.load(MODEL_FILES["cancer_scaler"])
    cancer_features = joblib.load(MODEL_FILES["cancer_features"])
    cancer = load_breast_cancer()
    X_c = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_c = cancer.target
    X_c_train, _, y_c_train, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=0)
    X_c_train_scaled = cancer_scaler.transform(X_c_train)

    models = {
        "diabetes": (diabetes_model, diabetes_scaler),
        "heart": (heart_model, heart_scaler),
        "cancer": (cancer_model, cancer_scaler, cancer_features)
    }

    train_data = {
        "diabetes": (X_d_train_scaled, y_d_train),
        "heart": (X_h_train_scaled, y_h_train),
        "cancer": (X_c_train_scaled, y_c_train)
    }

    return models, train_data

models, TRAIN_DATA = load_models()

def diabetes_form():
    st.subheader("\U0001FA78 Diabetes Prediction Input")
    fields = {
        "Pregnancies": (0, 20), "Glucose": (0, 200), "BloodPressure": (0, 140),
        "SkinThickness": (0, 100), "Insulin": (0, 900), "BMI": (0.0, 70.0),
        "DiabetesPedigreeFunction": (0.0, 2.5), "Age": (1, 100)
    }
    return [st.number_input(k, min_value=v[0], max_value=v[1], value=v[0],
                            format="%.2f" if isinstance(v[0], float) else "%d") for k, v in fields.items()]

def heart_form():
    st.subheader("\U0001F493 Heart Disease Prediction Input")
    fields = {
        "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (80, 200), "chol": (100, 600),
        "fbs": (0, 1), "restecg": (0, 2), "thalach": (60, 210), "exang": (0, 1),
        "oldpeak": (0.0, 6.0), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3)
    }
    return [st.number_input(k, min_value=v[0], max_value=v[1], value=v[0],
                            format="%.2f" if isinstance(v[0], float) else "%d") for k, v in fields.items()]

def cancer_form():
    st.subheader("\U0001F9EC Breast Cancer Prediction Input")
    _, _, features = models["cancer"]
    inputs = [st.number_input(f, format="%.2f") for f in features[:10]]
    return inputs + [0.0] * (len(features) - len(inputs))

def predict(model, scaler, input_data, labels):
    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    return labels[result]

def model_performance(model, X_train, y_train):
    y_pred = model.predict(X_train)
    return classification_report(y_train, y_pred)

disease = st.sidebar.selectbox("Choose Disease to Predict", ["Diabetes", "Heart Disease", "Breast Cancer"])

if disease == "Diabetes":
    input_data = diabetes_form()
    if st.button("\U0001F50D Predict Diabetes"):
        label = predict(*models["diabetes"], input_data, {0: "❌ Not Diabetic", 1: "✅ Diabetic"})
        st.success(f"Prediction: {label}")
        X_train, y_train = TRAIN_DATA["diabetes"]
        st.text(model_performance(models["diabetes"][0], X_train, y_train))

elif disease == "Heart Disease":
    input_data = heart_form()
    if st.button("\U0001F50D Predict Heart Disease"):
        label = predict(*models["heart"], input_data, {0: "✅ No Heart Disease", 1: "⚠️ Heart Disease Detected"})
        st.success(f"Prediction: {label}")
        X_train, y_train = TRAIN_DATA["heart"]
        st.text(model_performance(models["heart"][0], X_train, y_train))

elif disease == "Breast Cancer":
    input_data = cancer_form()
    model, scaler, _ = models["cancer"]
    if st.button("\U0001F50D Predict Breast Cancer"):
        label = predict(model, scaler, input_data, {0: "❌ Malignant", 1: "✅ Benign"})
        st.success(f"Prediction: {label}")
        X_train, y_train = TRAIN_DATA["cancer"]
        st.text(model_performance(model, X_train, y_train))
