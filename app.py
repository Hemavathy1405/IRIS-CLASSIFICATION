# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌸 Iris Flower Species Predictor")
st.markdown("Enter the flower's measurements below and get prediction.")

# Load and train model
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_scaled, y)

# User input
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.3)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

# Convert input to array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

# Predict class & probability
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# Output
st.subheader("🌿 Prediction Result:")
st.success(f"Predicted Species: **{target_names[prediction[0]]}**")

st.subheader("📊 Prediction Probabilities:")
for i, species in enumerate(target_names):
    st.write(f"{species}: **{probability[0][i]*100:.2f}%**")
