# Spam Email Classifier
## Project Overview
A web-based spam detection system that classifies emails/messages as SPAM or NOT SPAM with 97.49% accuracy.
## Live Demo
Run locally: `python app.py` then open `http://localhost:5000`
## Tech Stack
- Python, Flask (Backend)
- HTML, CSS (Frontend)
- Scikit-learn, TF-IDF, Logistic Regression (ML)
## Features
- Clean web interface
- Real-time prediction
- Confidence score displayed
- 97.49% accuracy
## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 97.49% |
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF |
| Dataset | SMS Spam Collection + Custom patterns |
## How to Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
