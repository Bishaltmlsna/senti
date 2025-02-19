import streamlit as st

# ✅ Move this line to the very top (before any other Streamlit functions)
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="wide")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and vectorizer using joblib
try:
    model = joblib.load("sentiment_analysis_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# Custom CSS for better UI
st.markdown("""
    <style>
        .positive {color: green; font-weight: bold; font-size: 20px;}
        .negative {color: red; font-weight: bold; font-size: 20px;}
        .header {font-size: 30px; color: #2C3E50; text-align: center; margin-top: 20px;}
        .subheader {font-size: 20px; color: #34495E; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='header'>💬 Sentiment Analysis with Confidence Scores</div>", unsafe_allow_html=True)
st.write("### Predict the sentiment of your text or upload a CSV file for bulk predictions.")

# ------------------------------------------------------------------
# 🔍 **Single Sentence Prediction**
# ------------------------------------------------------------------
st.markdown("<div class='subheader'>🔍 Single Sentence Prediction</div>", unsafe_allow_html=True)
user_input = st.text_area("Enter a sentence:", height=100)

if st.button("🚀 Predict Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            user_input_tfidf = vectorizer.transform([user_input])
            try:
                prediction = model.predict(user_input_tfidf)  # 0 or 1
                confidence = model.predict_proba(user_input_tfidf)  # Probabilities
            except Exception as e:
                st.error(f"❌ Error making predictions: {e}")
                st.stop()

            label_map = {1: "Positive 😀", 0: "Negative 😔"}
            sentiment = label_map[prediction[0]]
            confidence_score = confidence[0].max()

            color_class = "positive" if prediction[0] == 1 else "negative"
            st.markdown(f"<div class='{color_class}'>🧾 Sentiment: {sentiment}</div>", unsafe_allow_html=True)
            st.write(f"**📊 Confidence Score:** {confidence_score:.2f}")
    else:
        st.warning("⚠️ Please enter a valid sentence.")

# ------------------------------------------------------------------
# 📂 **Bulk Prediction from CSV**
# ------------------------------------------------------------------
st.markdown("<div class='subheader'>📂 Bulk Prediction from CSV File</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type=['csv'])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if 'text' not in data.columns:
            st.error("🚫 The uploaded file must contain a 'text' column.")
        else:
            with st.spinner("Processing uploaded file..."):
                text_data = data['text'].fillna("")
                text_tfidf = vectorizer.transform(text_data)

                try:
                    predictions = model.predict(text_tfidf)
                    confidence_scores = model.predict_proba(text_tfidf)
                except Exception as e:
                    st.error(f"❌ Error making batch predictions: {e}")
                    st.stop()

                label_map = {1: "Positive 😀", 0: "Negative 😔"}
                data['Sentiment'] = [label_map[pred] for pred in predictions]
                data['Confidence'] = confidence_scores.max(axis=1)

                st.write("### 📊 Prediction Results")
                st.dataframe(data[['text', 'Sentiment', 'Confidence']])

                # 📈 **Plot sentiment distribution**
                st.write("### 📈 Sentiment Distribution")
                sentiment_counts = data['Sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=["#2ecc71", "#e74c3c"])
                ax.axis('equal')
                st.pyplot(fig)

                # 📥 **Download results as CSV**
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
