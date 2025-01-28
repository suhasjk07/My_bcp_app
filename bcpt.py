import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from geopy.geocoders import Nominatim
import requests
from streamlit_chat import message  # For interactive chatbot-like features
import pyttsx3  # For real-time voice synthesis
import json  # To handle API responses

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()

# Function to call Gemini API
# Import Gemini generative AI library

# Configure Gemini API key
genai.configure(api_key="AIzaSyD8FGMpZVqRZv9B2KRUpzW4eViFr5SHEC8")  # Replace with your actual API key

# Function to call Gemini AI and generate responses
def call_gemini_api(prompt):
    try:
        # Create a generative model object
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)  # Generate content based on the prompt
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "I'm sorry, I couldn't process your request.")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Gemini API: {e}"

def load_dummy_data():
    # Simulate data for training
    data = {
        "Age": np.random.randint(20, 80, 1000),
        "BMI": np.random.uniform(18.5, 40, 1000),
        "Family_History": np.random.choice([0, 1], size=1000),
        "Lump": np.random.choice([0, 1], size=1000),
        "Skin_Changes": np.random.choice([0, 1], size=1000),
        "Breast_Pain": np.random.choice([0, 1], size=1000),
        "Risk": np.random.choice([0, 1], size=1000),
    }
    return pd.DataFrame(data)

def train_dummy_model():
    data = load_dummy_data()
    X = data.drop("Risk", axis=1)
    y = data["Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Load model and accuracy
dummy_model, model_accuracy = train_dummy_model()

def get_cancer_treatment_centers():
    # Fetch the top cancer treatment centers using a placeholder API or Google Maps link
    return [
        "Mayo Clinic, Rochester, USA",
        "MD Anderson Cancer Center, Houston, USA",
        "Royal Marsden Hospital, London, UK",
        "Memorial Sloan Kettering Cancer Center, New York, USA",
        "Apollo Hospitals, Chennai, India"
    ]

def speak_text(text):
    """Use text-to-speech to read out text."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    st.title("Breast Cancer Prediction and Assistance Tool")
    st.sidebar.header("User Input Features")

    # Sidebar for user input
    def user_input_features():
        age = st.sidebar.slider("Age", 20, 80, 30)
        bmi = st.sidebar.slider("BMI", 18.5, 40.0, 25.0)
        family_history = st.sidebar.selectbox("Family History of Breast Cancer", ("No", "Yes"))
        lump = st.sidebar.selectbox("Lump in Breast", ("No", "Yes"))
        skin_changes = st.sidebar.selectbox("Skin Changes", ("No", "Yes"))
        breast_pain = st.sidebar.selectbox("Breast Pain", ("No", "Yes"))

        data = {
            "Age": age,
            "BMI": bmi,
            "Family_History": 1 if family_history == "Yes" else 0,
            "Lump": 1 if lump == "Yes" else 0,
            "Skin_Changes": 1 if skin_changes == "Yes" else 0,
            "Breast_Pain": 1 if breast_pain == "Yes" else 0,
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Display user input
    st.subheader("User Input Features")
    st.write(input_df)

    # Predict risk using the dummy model
    st.subheader("Prediction")
    prediction = dummy_model.predict(input_df)[0]
    prediction_probability = dummy_model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result_text = f"High Risk of Breast Cancer (Probability: {prediction_probability:.2f})"
        st.error(result_text)
        st.info("Recommended Actions:")
        st.write("- Consult a specialist immediately.")
        st.write("- Schedule a mammogram or further screening tests.")
    else:
        result_text = f"Low Risk of Breast Cancer (Probability: {prediction_probability:.2f})"
        st.success(result_text)
        st.write("Continue regular self-checks and maintain a healthy lifestyle.")

    # Speak the result
    if st.button("Hear Result"):
        speak_text(result_text)

    # Additional sections
    st.subheader("Precautionary Tips")
    st.write("- Perform regular self-breast exams.")
    st.write("- Maintain a healthy weight and avoid processed foods.")
    st.write("- Exercise regularly and limit alcohol consumption.")
    st.write("- Stay informed about family medical history.")

    st.subheader("Healthy Food Recommendations")
    st.write("- Include leafy greens, berries, and cruciferous vegetables in your diet.")
    st.write("- Opt for whole grains, lean proteins, and healthy fats.")
    st.write("- Stay hydrated and avoid sugary beverages.")

    st.subheader("Top Cancer Treatment Centers")
    treatment_centers = get_cancer_treatment_centers()
    for center in treatment_centers:
        st.write(f"- {center}")

    # Add Google Maps link for treatment center location
    st.subheader("Find Treatment Centers on Google Maps")
    map_query = "https://www.google.com/maps/search/top+cancer+treatment+centers"
    st.markdown(f"[View on Google Maps]({map_query})")

    # Add patient booking feature
    st.subheader("Book an Appointment as a Patient")
    with st.form("appointment_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        date = st.date_input("Preferred Appointment Date")
        submit = st.form_submit_button("Book Appointment")

        if submit:
            st.success(f"Appointment successfully booked for {name} on {date}!")

    # Real-time Image Analysis Section
    st.subheader("Real-Time Image Analysis")
    uploaded_file = st.file_uploader("Upload a mammogram image for analysis:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image... (AI-based analysis will be integrated here)")

    # Interactive Chat Feature
    st.subheader("Interactive Chat Assistance")
    user_input = st.text_input("Ask a question about breast cancer:")
    if user_input:
        response = call_gemini_api(user_input)
        st.write(f"AI Response: {response}")
        # Speak the AI's response
        if st.button("Hear AI Response"):
            speak_text(response)

if __name__ == "__main__":
    main()
