import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = {
    'hours_studied': [2, 5, 1, 4, 6, 3, 7, 8, 9, 10],
    'sleep_hours': [6, 7, 5, 6, 8, 7, 7, 8, 9, 8],
    'attendance': [70, 85, 60, 90, 95, 80, 88, 92, 96, 98],
    'score': [50, 80, 40, 75, 92, 65, 85, 90, 95, 98]
}
df = pd.DataFrame(data)


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


