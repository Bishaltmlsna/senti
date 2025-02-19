import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💡", layout="wide")

# Load the model and vectorizer
try:
    model = joblib.load("sentiment_analysis_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Custom CSS for professional UI
st.markdown(
    """
    <style>
        body {background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif;}
        .main-container {padding: 50px; text-align: center;}
        .title {font-size: 38px; font-weight: bold; color: #2c3e50;}
        .subtitle {font-size: 20px; color: #34495e; margin-bottom: 30px;}
        .result-box {padding: 15px; border-radius: 8px; text-align: center; font-size: 22px; font-weight: bold; width: 50%; margin: auto;}
        .positive {background-color: #27ae60; color: white;}
        .negative {background-color: #c0392b; color: white;}
        .sidebar-title {font-size: 20px; font-weight: bold; color: #2980b9; text-align: center;}
        .upload-section {padding: 15px; border: 2px solid #2980b9; text-align: center; background-color: #ecf0f1; border-radius: 10px;}
        .stButton > button {background-color: #1abc9c; color: white; font-size: 18px; padding: 10px 20px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='title'>💡 Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ App Controls")
st.sidebar.info("Use this dashboard to analyze the sentiment of text data effectively.")

# Tabs for navigation
selected_tab = st.sidebar.radio("Navigation", ["📄 Single Text Analysis", "📊 Bulk Analysis (CSV)"])

# Single Sentence Prediction
if selected_tab == "📄 Single Text Analysis":
    st.subheader("🔍 Enter a sentence to analyze:")
    user_input = st.text_area("", height=100)
    if st.button("🚀 Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                input_tfidf = vectorizer.transform([user_input])
                prediction = model.predict(input_tfidf)[0]
                confidence = model.predict_proba(input_tfidf)[0].max()

                sentiment_label = "Positive 😀" if prediction == 1 else "Negative 😔"
                color_class = "positive" if prediction == 1 else "negative"

                st.markdown(
                    f"<div class='result-box {color_class}'>🧾 Sentiment: {sentiment_label}</div>",
                    unsafe_allow_html=True,
                )
                st.write(f"**📊 Confidence Score:** {confidence:.2f}")
        else:
            st.warning("⚠️ Please enter a valid sentence.")

# Bulk Prediction from CSV
if selected_tab == "📊 Bulk Analysis (CSV)":
    st.subheader("📂 Upload a CSV File")
    st.markdown("<div class='upload-section'>Drag & Drop your CSV file here</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            if 'text' not in data.columns:
                st.error("🚫 The uploaded file must contain a 'text' column.")
            else:
                with st.spinner("Processing file..."):
                    text_data = data['text'].fillna("")
                    text_tfidf = vectorizer.transform(text_data)
                    predictions = model.predict(text_tfidf)
                    confidence_scores = model.predict_proba(text_tfidf).max(axis=1)

                    data['Sentiment'] = ["Positive 😀" if pred == 1 else "Negative 😔" for pred in predictions]
                    data['Confidence'] = confidence_scores

                    st.write("### 📊 Prediction Results")
                    st.dataframe(data[['text', 'Sentiment', 'Confidence']])

                    # 📈 Sentiment Distribution
                    st.write("### 📈 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(y=data['Sentiment'], palette=["#27ae60", "#c0392b"], ax=ax)
                    ax.set_title("Sentiment Breakdown")
                    st.pyplot(fig)

                    # 📥 Download results as CSV
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV Results",
                        data=csv,
                        file_name='sentiment_analysis_results.csv',
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")




          
