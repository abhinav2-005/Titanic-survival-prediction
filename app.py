import streamlit as st
import pandas as pd
import joblib


# MUST be the first Streamlit command
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)


# -----------------------------
# Load saved files
# -----------------------------
@st.cache_resource
def load_files():
    model = joblib.load("model.pkl")
    scaler = joblib.load("standerd_scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


model, scaler, columns = load_files()


# -----------------------------
# Frontend
# -----------------------------
st.title("Titanic Survival Prediction")

st.write(
    "Enter the passenger details to predict whether the passenger survived."
)

st.subheader("Passenger Details")


pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

alone = st.selectbox(
    "Travelling Alone?",
    ["Yes", "No"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    [
        "Cherbourg (C)",
        "Queenstown (Q)",
        "Southampton (S)"
    ]
)


# -----------------------------
# Encoding
# -----------------------------
sex_value = 1 if sex == "Male" else 0

alone_value = 1 if alone == "Yes" else 0

embarked_C = 1 if embarked == "Cherbourg (C)" else 0
embarked_Q = 1 if embarked == "Queenstown (Q)" else 0
embarked_S = 1 if embarked == "Southampton (S)" else 0


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):

    input_data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex_value,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "alone": alone_value,
        "embarked_C": embarked_C,
        "embarked_Q": embarked_Q,
        "embarked_S": embarked_S
    }])

    feature_columns = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "alone",
        "embarked_C",
        "embarked_Q",
        "embarked_S"
    ]

    input_data = input_data[feature_columns]

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("Passenger is predicted to survive!")
    else:
        st.error("Passenger is predicted not to survive!")

    # Show probability if SVC supports it
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1]

        st.info(
            f"Survival Probability: **{probability * 100:.2f}%**"
        )