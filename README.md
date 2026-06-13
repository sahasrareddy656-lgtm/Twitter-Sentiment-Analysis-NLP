# Twitter Sentiment Analysis using NLP and Machine Learning

## Overview

This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. The model classifies tweets into three categories:

* Positive 😊
* Neutral 😐
* Negative 😞

An interactive Streamlit application allows users to enter tweet text and instantly predict its sentiment.

## Dataset

Dataset: Twitter Sentiment Dataset

* Total Records: 162,980 Tweets
* Features:

  * clean_text
  * category

Sentiment Labels:

* 1 = Positive
* 0 = Neutral
* -1 = Negative

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression
* Streamlit

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Text Vectorization using TF-IDF
5. Train-Test Split
6. Logistic Regression Model Training
7. Model Evaluation
8. Streamlit Dashboard Development

## Model Performance

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer
* Accuracy: 85.51%

## Features

* Real-Time Tweet Sentiment Prediction
* Three-Class Classification
* NLP-Based Text Processing
* Interactive Streamlit Dashboard
* Machine Learning Model Deployment

## Example Predictions

Input:
I absolutely love this new smartphone. Amazing battery life.

Output:
Positive 😊

Input:
The meeting is scheduled for tomorrow at 10 AM.

Output:
Neutral 😐

Input:
This product is terrible and a complete waste of money.

Output:
Negative 😞

## Project Structure

Twitter-Sentiment-Analysis-NLP/

├── data/

├── screenshots/

├── src/

│   └── train_model.py

├── app.py

├── model.pkl

├── vectorizer.pkl

├── requirements.txt

├── README.md

└── .gitignore

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Screenshots

### Home Page

(Add Screenshot Here)

### Positive Prediction

(Add Screenshot Here)

### Negative Prediction

(Add Screenshot Here)

## Future Improvements

* Deep Learning Models (LSTM/BERT)
* Social Media Trend Analysis
* Sentiment Visualization Dashboard
* Real-Time Twitter API Integration

## Author

Sahasra Reddy
