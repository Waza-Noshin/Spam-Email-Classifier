import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("SPAM EMAIL CLASSIFIER - TRAINING WITH COMPLETE DATASET")
print("=" * 60)

# Load dataset from local CSV
print("\n📥 Loading dataset from spam_dataset.csv...")

try:
    df = pd.read_csv('spam_dataset.csv')
    print(f"✅ Dataset loaded: {len(df)} messages")
    print(f"   Spam: {(df['label'] == 'spam').sum()}")
    print(f"   Ham: {(df['label'] == 'ham').sum()}")
except FileNotFoundError:
    print("❌ spam_dataset.csv not found! Please make sure the file is in the same folder.")
    exit()

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n🔄 Cleaning messages...")
df['clean_message'] = df['message'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer
print("🔄 Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
print("🔄 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   ✅ Accuracy: {accuracy * 100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Critical test
print("\n" + "=" * 60)
print("CRITICAL TEST - Bank Verification Messages")
print("=" * 60)

critical_tests = [
    "Bank account verification required. Click here",
    "Your bank account has been locked. Verify now",
    "URGENT: Your account has been locked. Verify your details immediately",
    "Congratulations! You've won a $1000 gift card",
    "FREE MONEY! Earn $5000 per week",
    "Hi John, can you review the attached document",
]

for msg in critical_tests:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    result = "✅ SPAM" if pred == 1 else "❌ NOT SPAM"
    print(f"   {result} (Confidence: {prob*100:.1f}%) - {msg[:50]}")

# Save model
print("\n💾 Saving model...")
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved successfully!")
print("=" * 60)