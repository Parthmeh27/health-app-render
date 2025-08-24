import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load model
model_path = "ML_model.sav"
loaded_model = pickle.load(open(model_path, "rb"))

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Sleep_health.csv")

df = load_data()

# ----------------- Prediction Helpers -----------------
def generate_synthetic_data(num_samples):
    np.random.seed(0)
    data = {
        'Age': np.random.uniform(20, 70, num_samples),
        'Sleep_Hours': np.random.uniform(4, 10, num_samples),
        'Physical_Activity_Level': np.random.uniform(1, 10, num_samples),
        'Heart_Rate': np.random.uniform(60, 100, num_samples),
        'Daily_Steps': np.random.uniform(1000, 15000, num_samples)
    }
    df = pd.DataFrame(data)
    return df

def predict_stress_level(input_data):
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

# ----------------- Streamlit App -----------------
st.title("🧠 Sleep & Stress Health Dashboard")

tab1, tab2 = st.tabs(["📊 Data Analysis", "🤖 Stress Prediction"])

# ----------------- TAB 1: EDA -----------------
with tab1:
    st.header("Dataset Overview")
    st.write(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Sleep Hours Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Sleep_Hours'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Interactive Plots")
    x_axis = st.selectbox("Select X-axis", df.columns, index=0)
    y_axis = st.selectbox("Select Y-axis", df.columns, index=1)
    fig = px.scatter(df, x=x_axis, y=y_axis, color="Stress_Level" if "Stress_Level" in df.columns else None)
    st.plotly_chart(fig)

# ----------------- TAB 2: Prediction -----------------
with tab2:
    st.header("Generate Synthetic Data and Predict Stress Levels")
    num_samples = st.number_input("Enter number of synthetic samples", min_value=1, max_value=1000, value=100)

    if st.button("Generate and Predict"):
        data = generate_synthetic_data(num_samples)
        st.write("Synthetic Data Sample:", data.head())

        data['Predicted_Stress_Level'] = data.apply(lambda row: predict_stress_level(row), axis=1)
        st.write("Predictions for Synthetic Data:", data.head())

        mean_prediction = data['Predicted_Stress_Level'].mean()
        st.success(f"Mean Predicted Stress Level: {mean_prediction:.2f}")

    st.header("Manual Input Prediction")
    Age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
    Sleep_Hours = st.number_input("Enter Sleep Hours", min_value=0.0, max_value=15.0, value=7.0)
    Physical_Activity_Level = st.number_input("Enter Physical Activity Level (1-10)", min_value=1.0, max_value=10.0, value=5.0)
    Heart_Rate = st.number_input("Enter Heart Rate", min_value=40.0, max_value=200.0, value=80.0)
    Daily_Steps = st.number_input("Enter Daily Steps", min_value=0, max_value=50000, value=5000)

    if st.button("Predict Stress Level"):
        diagnosis = predict_stress_level([Age, Sleep_Hours, Physical_Activity_Level, Heart_Rate, Daily_Steps])
        st.success(f"Predicted Stress Level: {diagnosis}")
