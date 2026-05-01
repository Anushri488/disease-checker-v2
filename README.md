# 🩺 SymptomAI — Disease Symptom Checker

An ML-powered full-stack web application that predicts diseases based on user-selected symptoms using a trained Random Forest classifier.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-99.5%25-brightgreen?style=flat-square)

---

## ✨ Features

- 🔬 **ML Prediction** — Trained Random Forest model with **99.5% accuracy**
- 🦠 **51 Diseases** across **141 symptoms**
- 📊 **Confidence Scoring** — Shows prediction confidence percentage
- 🏆 **Top 5 Differential Diagnosis** — Lists other possible diseases
- 🔴 **Severity Classification** — Mild / Moderate / Severe
- ⚠️ **Doctor Urgency Warning** — Tells when to see a doctor immediately
- 📋 **Prediction History** — Saves past predictions
- 📄 **PDF Report Generation** — Download a full diagnosis report
- 🔍 **Symptom Search & Filters** — Category-wise symptom browsing

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn (Random Forest, Decision Tree, Naive Bayes) |
| Backend | Python, Flask, Flask-CORS |
| Data Processing | Pandas, NumPy, Joblib |
| PDF Generation | ReportLab |
| Frontend | React 18, Vite, Axios |
| Deployment | Render (Backend) + Netlify (Frontend) |

---

## 📁 Project Structure

```
disease-checker/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── train.py            # ML model training script
│   ├── model.pkl           # Trained Random Forest model
│   ├── encoder.pkl         # Label encoder
│   ├── symptoms_list.pkl   # All symptoms list
│   ├── disease_info.pkl    # Disease metadata
│   ├── requirements.txt    # Python dependencies
│   └── Procfile            # For Render deployment
└── frontend/
    ├── src/
    │   ├── App.jsx         # Main React component
    │   └── main.jsx        # Entry point
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anushri488/disease-checker-v2.git
cd disease-checker-v2/disease-checker
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python train.py        # Train the model (run once)
python app.py          # Start Flask API on :5000
```

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm run dev            # Start React app on :5173
```

### 4. Open in Browser

```
http://localhost:5173
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check + model stats |
| GET | `/symptoms` | List all 141 symptoms |
| GET | `/diseases` | List all diseases with metadata |
| POST | `/predict` | Predict disease from symptoms |
| GET | `/history` | Get prediction history |
| DELETE | `/history` | Clear prediction history |
| POST | `/report` | Generate PDF report |

### POST `/predict` — Example

**Request:**
```json
{
  "symptoms": ["fatigue", "high_fever", "chills", "headache", "muscle_pain"]
}
```

**Response:**
```json
{
  "disease": "Malaria",
  "confidence": 94.2,
  "severity": "severe",
  "doctor_urgency": "high",
  "specialist": "Infectious Disease Specialist",
  "description": "Mosquito-borne parasitic infection...",
  "top5": [
    { "disease": "Malaria", "confidence": 94.2 },
    { "disease": "Dengue", "confidence": 3.8 },
    { "disease": "Typhoid", "confidence": 1.2 }
  ],
  "matched_symptoms": ["fatigue", "high_fever", "chills", "headache", "muscle_pain"]
}
```

---

## 🤖 ML Model Details

| Model | Test Accuracy |
|-------|--------------|
| **Random Forest** ✅ | **99.5%** |
| Decision Tree | 75.9% |
| Naive Bayes | 43.4% |

- **Dataset:** Synthetically generated from clinical symptom mappings
- **Training samples:** 3,060 (60 per disease)
- **Cross-validation score:** 99.4% (5-fold)
- **Features:** Binary symptom encoding (1 = present, 0 = absent)

---

## ☁️ Deployment

### Backend → Render
1. Connect GitHub repo on [render.com](https://render.com)
2. Root Directory: `disease-checker/backend`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`

### Frontend → Netlify
1. Connect GitHub repo on [netlify.com](https://netlify.com)
2. Base Directory: `disease-checker/frontend`
3. Build Command: `npm run build`
4. Publish Directory: `disease-checker/frontend/dist`

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified doctor for any health concerns.

---

## 👩‍💻 Author

**Anushri** — [GitHub](https://github.com/Anushri488)
