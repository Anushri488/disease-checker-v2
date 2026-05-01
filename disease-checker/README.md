# 🩺 SymptomAI — Disease Checker v2

AI-powered disease prediction app using Random Forest ML model.

## ✨ Features

- **42 diseases** predicted (up from 37)
- **110 symptoms** with severity labels (Mild / Moderate / Severe)
- **Disease description** — what the condition is
- **Doctor visit warning** — when to seek medical help
- **Emergency alert** — red banner for life-threatening conditions
- **Precautions & self-care** tips per disease
- **Prediction history** — last 20 predictions saved locally
- **PDF report download** — detailed, formatted medical-style report
- **Category filters** — Skin, Digestive, Fever, Pain, Respiratory, Neuro, etc.

## 🧠 ML Model

| | Old | New |
|---|---|---|
| Algorithm | Decision Tree | **Random Forest (200 trees)** |
| Diseases | 37 | **42** |
| Accuracy | ~74% | **99.67% CV** |
| Dataset | 50 samples/disease | **100 samples/disease** |

## 🚀 Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python train.py        # retrain model (already done)
python app.py          # starts Flask on port 5000
```

### Frontend
```bash
cd frontend
npm install
npm run dev            # starts Vite on port 5173
```

## ⚠️ Disclaimer
This is an educational ML project. Not a substitute for professional medical advice.
