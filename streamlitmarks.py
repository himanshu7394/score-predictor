import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data.csv")

st.title("🎯 Student Score Predictor")
st.write("This app predicts student grades based on study hours, sleep hours, and attendance.")

# Show dataset preview
with st.expander("📊 See Dataset"):
    st.dataframe(df.head(20))

# -------------------------------
# Train Model
# -------------------------------
X = df[['study_hours','sleep_hours','attendance']]
y = df['grades']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write(f"**Intercept:** {model.intercept_:.2f}")
st.write(f"**Coefficients:** {model.coef_}")
st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

# Plot Actual vs Predicted
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("Actual Grades")
ax.set_ylabel("Predicted Grades")
ax.set_title("Actual vs Predicted Grades")
st.pyplot(fig)

# -------------------------------
# User Input for Prediction
# -------------------------------
st.subheader("🔮 Predict Your Grade")

study_hours = st.number_input("📚 Study Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
sleep_hours = st.number_input("😴 Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("🏫 Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    input_data = np.array([[study_hours, sleep_hours, attendance]])
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Grade: {prediction[0]:.2f}")