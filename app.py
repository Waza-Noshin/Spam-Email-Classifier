from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
import os

# Download stopwords once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load model
print("Loading model...")
with open('models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print("Model loaded!")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_spam(message):
    cleaned = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    
    return {
        'is_spam': bool(prediction),
        'confidence': float(max(probability)),
        'message': message
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    result = predict_spam(message)
    return jsonify(result)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True, port=5000)