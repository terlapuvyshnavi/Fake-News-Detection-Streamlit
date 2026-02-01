fake neus detection.py save code with this name
#code

# ==============================
# Fake News Detection System
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# 2. Load Dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

# 3. Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# 4. Split Dataset
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train Machine Learning Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test_tfidf)

print("\n📊 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Prediction on New Input
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "REAL NEWS 🟢" if prediction[0] == 1 else "FAKE NEWS 🔴"

# Test with sample input
sample_news = input("\n📰 Enter news text to check authenticity:\n")
print("\nPrediction:", predict_news(sample_news))


the below code is for the app

say the code as app.py

# ==============================
# Fake News Detection Web App
# ==============================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Streamlit UI - File Upload
# --------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")
st.write("### Upload CSVs with news articles to classify")

uploaded_fake = st.file_uploader("Upload Fake News CSV", type="csv")
uploaded_true = st.file_uploader("Upload Real News CSV", type="csv")

# --------------------------
# Only run after both files are uploaded
# --------------------------
if uploaded_fake is not None and uploaded_true is not None:
    try:
        fake = pd.read_csv(uploaded_fake)
        true = pd.read_csv(uploaded_true)

        # Automatically detect news text column
        text_col = None
        for col in fake.columns:
            if 'text' in col.lower() or 'content' in col.lower():
                text_col = col
                break

        if text_col is None:
            st.error("Could not find a text column in your CSV. Please check column names.")
        else:
            fake["label"] = 0
            true["label"] = 1

            df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

            # Text preprocessing
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                text = str(text)
                text = re.sub(r'[^a-zA-Z]', ' ', text)
                text = text.lower()
                words = text.split()
                words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                return " ".join(words)

            df["clean_text"] = df[text_col].apply(clean_text)

            # Train ML Model
            X = df["clean_text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            model = LogisticRegression()
            model.fit(X_train_tfidf, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

            # Prediction Input
            news_text = st.text_area("Enter news text to classify:", height=200)
            if st.button("🔍 Predict"):
                if news_text.strip() == "":
                    st.warning("Please enter some news text.")
                else:
                    cleaned = clean_text(news_text)
                    vect = vectorizer.transform([cleaned])
                    pred = model.predict(vect)[0]
                    if pred == 1:
                        st.success("✅ This looks like REAL NEWS")
                    else:
                        st.error("🚨 This looks like FAKE NEWS")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

