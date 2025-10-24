import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


import numpy as np

np.random.seed(42)  # for reproducibility

n = 50  # number of samples
hours_studied = np.random.randint(0, 11, n)      # 0–10 hours
sleep_hours = np.random.randint(4, 11, n)        # 4–10 hours
attendance = np.random.randint(50, 101, n)       # 50%–100%

# realistic score calculation with some noise
score = (
    hours_studied * 7       # 1 extra hour adds ~7 points
    + sleep_hours * 2       # 1 extra hour sleep adds ~2 points
    + (attendance - 50) * 0.3  # attendance contributes moderately
    + np.random.normal(0, 5, n)  # add noise
)

# clip scores to 0–100
score = np.clip(score, 0, 100)

df = pd.DataFrame({
    'hours_studied': hours_studied,
    'sleep_hours': sleep_hours,
    'attendance': attendance,
    'score': score.round(1)
})



X = df[['hours_studied', 'sleep_hours', 'attendance']]
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


st.title("📘 Student Score Predictor")
st.write("Predict your exam score based on your study habits!")


hours = st.slider("Hours Studied", 0, 12, 5)
sleep = st.slider("Hours of Sleep", 0, 12, 7)
attendance = st.slider("Attendance (%)", 50, 100, 85)


if st.button("Predict Score"):
    predicted = model.predict([[hours, sleep, attendance]])[0]
    st.success(f"🎯 Predicted Score: {predicted:.2f}")


with st.expander("See Training Data"):
    st.dataframe(df)



st.sidebar.markdown("---")  
st.sidebar.header("About the Creator")
st.sidebar.write("👤 Name: Fatin Ilham")
st.sidebar.write("📸 Instagram: [@spiritofhonestyy](https://www.instagram.com/spiritofhonestyyy/)")
st.sidebar.write("📘 Facebook: [Fatin Ilham](https://www.facebook.com/profile.php?id=61572732399921)")
st.sidebar.write("💻 GitHub: [Fatin's GitHub](https://github.com/fatin-ilham)")
st.sidebar.write("📧 Email: fatin.ilham@g.bracu.ac.bd")



