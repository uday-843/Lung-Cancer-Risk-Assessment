# Lung Cancer Risk Assessment Tool

A web-based, AI-powered application to evaluate lung cancer risk based on user-provided health factors. This tool collects 15 inputs via an interactive form, runs a Random Forest classifier in real time, and displays results—including probability, top contributing factors, and model performance metrics—through rich visualizations and recommendations.

---
🌐 **Live Demo**: [Lung Cancer Risk Assessment Tool](https://lung-cancer-risk-assessment.onrender.com)

---
## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture & File Structure](#architecture--file-structure)
4. [Technology Stack](#technology-stack)
5. [Installation & Setup](#installation--setup)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [How It Works](#how-it-works)
9. [Screenshots](#screenshots)
10. [Contributing](#contributing)
11. [License](#license)
12. [Disclaimer](#disclaimer)

---

## 🔍 Overview

The **Lung Cancer Risk Assessment Tool** uses a lightweight Flask backend and a React-inspired frontend to:

* **Collect**: Ask users for 15 critical health parameters (gender, age, smoking status, symptoms, lifestyle factors).
* **Predict**: Run a pretrained Random Forest model that outputs a lung cancer risk probability and “Yes/No” classification.
* **Visualize**: Show model performance (accuracy, precision, recall, F1-score), feature importance bars, a color-coded risk meter, and a list of top contributing factors.
* **Recommend**: Provide personalized medical guidance based on risk category (Low/High).

All user inputs are validated in the browser before submission. The frontend calls Flask endpoints to fetch metrics, feature importances, and predictions via JSON. The entire UI is responsive and mobile-first.

---

## ✨ Key Features

* **Comprehensive Questionnaire**

  * Collects 15 discrete health inputs: gender, age, smoking, symptoms, lifestyle, and chronic conditions.
  * Client-side validation ensures completeness and value bounds.

* **Real-Time Prediction**

  * Random Forest classifier served from Flask.
  * Returns risk label ("YES"/"NO") and probability scores instantly.

* **Detailed Model Insights**

  * Exposes endpoints to retrieve training metrics and feature importances.

* **Interactive Frontend Visualizations**

  * Risk Meter, Bar Charts (Chart.js), Feature Highlight Cards.

* **Responsive, Modern UI**

  * Built with HTML5, CSS3, and Vanilla JS.
  * Clean, medical-themed design optimized for all devices.

* **Lightweight Deployment**

  * Single-page HTML + Flask backend. No heavy frameworks or build step required.

---

## 📂 Architecture & File Structure

```
Lung-Cancer-Risk-Assessment/
├── app.py
├── requirements.txt
├── data/
│   └── survey-lung-cancer.csv
├── saved_models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── metrics.pkl
├── templates/
│   └── index.html
├── train_model.py
├── LICENSE
└── README.md
```

---

## 🔧 Technology Stack

* **Backend**: Python, Flask, scikit-learn, pandas, joblib
* **Frontend**: HTML5, CSS3, JavaScript, Chart.js, Font Awesome
* **Modeling**: Random Forest, GridSearchCV, Label Encoding, StandardScaler

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/218r1a7230/Lung-Cancer-Risk-Assessment.git
cd Lung-Cancer-Risk-Assessment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

To retrain the model (optional):

```bash
python train_model.py
```

---

## ▶️ Running the Application

```bash
python app.py
```

Then open your browser at: `http://127.0.0.1:5000/`

---

## 🔍 API Endpoints

* `GET /` → Serves HTML UI
* `POST /predict` → Returns prediction, probability, and encoded features
* `GET /metrics` → Returns model metrics JSON
* `GET /feature_importance` → Returns feature importance scores
* `GET /health` → Health check

---

## ✨ How It Works

* Inputs are encoded and scaled using saved encoders/scalers
* Model predicts class and probability
* Frontend displays:

  * Risk percentage
  * Risk classification (YES/NO)
  * Visual charts (risk meter, feature importances, metrics)
  * Top contributing health factors
  * Personalized recommendations

---

## 🖼️ Screenshots

![Screenshot 2025-06-03 215247](https://github.com/user-attachments/assets/bec23f44-6a09-40cb-9d7d-53b7f86d68ca)

![image](https://github.com/user-attachments/assets/f1294d65-4690-44b2-96e4-528d2549e09e)


---

## 👍 Contributing

1. Fork this repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your message"`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

---

## 📜 License

MIT License. See the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is intended for informational purposes only and **does not constitute medical advice**. Always consult a qualified healthcare provider for diagnosis or treatment.
