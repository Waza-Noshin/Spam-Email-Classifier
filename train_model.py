import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create models folder
os.makedirs('models', exist_ok=True)

print("=" * 50)
print("SPAM EMAIL CLASSIFIER - TRAINING")
print("=" * 50)

# Download dataset
print("\n📥 Downloading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"✅ Dataset loaded: {df.shape[0]} messages")

# Add custom spam patterns to improve detection
print("\n🔄 Adding custom spam patterns...")

custom_spam = pd.DataFrame([
    ['spam', 'URGENT: Your account has been locked. Verify now.'],
    ['spam', 'Your account is locked. Please click here to unlock.'],
    ['spam', 'ACCOUNT LOCKED! Immediate action required.'],
    ['spam', 'Security alert: Your account has been compromised.'],
    ['spam', 'Verify your account now or it will be suspended.'],
    ['spam', 'Your PayPal account is limited. Confirm your details.'],
    ['spam', 'Bank account verification required. Click here.'],
    ['spam', 'Your Amazon account has been locked due to suspicious activity.'],
    ['spam', 'IMPORTANT: Your account will be closed if you do not respond.'],
    ['spam', 'Security update required. Login immediately.'],
    ['spam', 'Your account has been suspended. Click to restore.'],
    ['spam', 'FINAL WARNING: Verify your account within 24 hours.'],
    ['spam', 'Your Apple ID has been locked. Verify now.'],
    ['spam', 'Netflix: Your account is on hold. Update payment.'],
    ['spam', 'Your Google account has been compromised. Secure it now.'],
], columns=['label', 'message'])

df = pd.concat([df, custom_spam], ignore_index=True)

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

spam_count = df['label'].sum()
ham_count = len(df) - spam_count

print(f"   Total messages: {len(df)}")
print(f"   Ham messages: {ham_count}")
print(f"   Spam messages: {spam_count}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

print("\n🔄 Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("🔄 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model
print("\n💾 Saving model...")
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved to 'models/' folder")
print("=" * 50)