import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import speech_recognition as sr
import pyttsx3
import time
import google.generativeai as genai  # Gemini API

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set page configuration for a modern UI
st.set_page_config(
    page_title="Breast Cancer Prediction Chat Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyD8FGMpZVqRZv9B2KRUpzW4eViFr5SHEC8"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('breast_cancer_data.csv')  # Replace with your dataset
    return data

def preprocess_data(data):
    data = data.dropna()
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Gemini AI Assistant
def gemini_response(user_input):
    try:
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Prediction function
def predict_cancer(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Voice input function
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I did not understand that."
        except sr.RequestError:
            return "Sorry, the speech service is down."

# Text-to-speech function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Advanced UI with Streamlit
def main():
    st.title("🩺 Breast Cancer Prediction Chat Assistant")
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px;
        }
        .stButton>button {
            border-radius: 20px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for additional features
    with st.sidebar:
        st.header("⚙️ Settings")
        # Dark/light mode toggle
        dark_mode = st.checkbox("Dark Mode", value=False)
        if dark_mode:
            st.markdown("""
                <style>
                .stApp {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                .stApp {
                    background-color: #ffffff;
                    color: #000000;
                }
                </style>
            """, unsafe_allow_html=True)

        # Voice assistant toggle
        voice_assistant = st.checkbox("Enable Voice Assistant", value=False)

    # Load data and train model
    data = load_data()
    X, y = preprocess_data(data)
    model, accuracy = train_model(X, y)

    # Main chat interface
    st.header("💬 Chat with the Assistant")
    user_input = ""

    if voice_assistant:
        if st.button("🎤 Speak"):
            user_input = voice_input()
            st.text_area("You :", value=user_input, height=50, max_chars=None, key=None)
    else:
        user_input = st.text_input("You Ask questions like predict, visualize, symptom, food to see advanced responses: ", "", placeholder="Type your message here...")

    if user_input:
        if "predict" in user_input.lower():
            st.info("Please enter the following details for prediction:")
            input_data = []
            for feature in X.columns:
                value = st.number_input(f"Enter {feature}", value=0.0)
                input_data.append(value)
            
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    time.sleep(2)  # Simulate prediction delay
                    input_data = np.array([input_data])
                    prediction = predict_cancer(model, input_data)
                    if prediction[0] == 1:
                        st.error("Prediction: High risk of breast cancer. Please consult a doctor.")
                        if voice_assistant:
                            speak("Prediction: High risk of breast cancer. Please consult a doctor.")
                    else:
                        st.success("Prediction: Low risk of breast cancer. Stay healthy!")
                        if voice_assistant:
                            speak("Prediction: Low risk of breast cancer. Stay healthy!")

        elif "visualize" in user_input.lower():
            st.header("📊 Data Visualizations")
            st.write("### Feature Distribution")
            selected_feature = st.selectbox("Select a feature to visualize", X.columns)
            fig, ax = plt.subplots()
            sns.histplot(data[selected_feature], kde=True, ax=ax)
            st.pyplot(fig)

            st.write("### Correlation Heatmap")
            corr = data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.write("### Pair Plot")
            pair_plot = sns.pairplot(data, hue="target")
            st.pyplot(pair_plot)

        elif "symptoms" in user_input.lower():
            st.header("🩺 Symptom Analyzer")
            symptoms = st.text_area("Enter your symptoms (comma-separated):")
            if st.button("Analyze Symptoms"):
                response = gemini_response(f"Analyze these symptoms: {symptoms}")
                st.write(response)
                if voice_assistant:
                    speak(response)

        elif "food" in user_input.lower():
            st.header("🍎 Food Recommendations")
            condition = st.text_input("Enter your health condition or symptoms:")
            if st.button("Get Food Recommendations"):
                response = gemini_response(f"Suggest foods for this condition: {condition}")
                st.write(response)
                if voice_assistant:
                    speak(response)

        else:
            response = gemini_response(user_input)
            st.text_area("Assistant: ", value=response, height=100, max_chars=None, key=None)
            if voice_assistant:
                speak(response)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ by Suhas J.K using Streamlit.")

# Corrected entry point
if __name__ == "__main__":
    main()