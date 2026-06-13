import streamlit as st
import pickle

# Load Model and Vectorizer
model = pickle.load(open("../model.pkl", "rb"))
vectorizer = pickle.load(open("../vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Analysis")

st.markdown("""
Analyze tweet sentiment using Machine Learning and NLP.
""")

tweet = st.text_area(
    "Enter Tweet Text",
    height=150
)

if st.button("Analyze Sentiment"):

    tweet_vector = vectorizer.transform([tweet])

    prediction = model.predict(tweet_vector)[0]

    if prediction == 1:
        st.success("😊 Positive Sentiment")

    elif prediction == 0:
        st.info("😐 Neutral Sentiment")

    else:
        st.error("😞 Negative Sentiment")