import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ✅ Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ✅ Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ✅ Load dataset
df = pd.read_csv("data.csv")

# ✅ Clean dataset
df["cleaned"] = df["text"].apply(clean_text)

# ✅ Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    ngram_range=(1, 2),
    max_features=1000
)

X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# ✅ Train model
model = MultinomialNB()
model.fit(X, y)

# ✅ NEW: Predict probabilities
def predict_sentiment_proba(text):
    text = clean_text(text)
    vect = vectorizer.transform([text])

    probs = model.predict_proba(vect)[0]
    labels = model.classes_

    return dict(zip(labels, probs))
