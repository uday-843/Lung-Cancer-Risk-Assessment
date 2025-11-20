from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging
import traceback
import json

from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Add this line

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_model():
    try:
        data = pd.read_csv('data/survey-lung-cancer.csv')
        data.columns = data.columns.str.strip().str.replace(' ', '_').str.upper()

        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype == 'object':
                    data[col].fillna(data[col].mode()[0], inplace=True)
                else:
                    data[col].fillna(data[col].median(), inplace=True)

        label_encoders = {}
        for col in data.columns:
            if col != 'LUNG_CANCER' and data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

        X = data.drop('LUNG_CANCER', axis=1)
        y = data['LUNG_CANCER']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        joblib.dump(best_model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump({
            'accuracy': accuracy,
            'classification_report': clf_report,
            'confusion_matrix': conf_matrix,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }, 'metrics.pkl')

        logging.info("Model trained and artifacts saved.")
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        traceback.print_exc()

if not os.path.exists('model.pkl'):
    logging.info("Model not found. Training model...")
    train_model()

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    metrics = joblib.load('metrics.pkl')
    logging.info("Artifacts loaded successfully.")
except Exception as e:
    logging.error(f"Error loading artifacts: {str(e)}")
    model = None
    scaler = None
    label_encoders = None
    metrics = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        required_fields = [
            'gender', 'age', 'smoking', 'yellow_fingers', 'anxiety',
            'peer_pressure', 'chronic_disease', 'fatigue', 'allergy',
            'wheezing', 'alcohol_consuming', 'coughing',
            'shortness_of_breath', 'swallowing_difficulty', 'chest_pain'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = pd.DataFrame([{
            'GENDER': data['gender'],
            'AGE': data['age'],
            'SMOKING': data['smoking'],
            'YELLOW_FINGERS': data['yellow_fingers'],
            'ANXIETY': data['anxiety'],
            'PEER_PRESSURE': data['peer_pressure'],
            'CHRONIC_DISEASE': data['chronic_disease'],
            'FATIGUE': data['fatigue'],
            'ALLERGY': data['allergy'],
            'WHEEZING': data['wheezing'],
            'ALCOHOL_CONSUMING': data['alcohol_consuming'],
            'COUGHING': data['coughing'],
            'SHORTNESS_OF_BREATH': data['shortness_of_breath'],
            'SWALLOWING_DIFFICULTY': data['swallowing_difficulty'],
            'CHEST_PAIN': data['chest_pain']
        }])

        for col in input_data.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_data[col] = input_data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                )

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

        response = {
            'prediction': (label_encoders['LUNG_CANCER'].inverse_transform(prediction)[0] 
                          if 'LUNG_CANCER' in label_encoders else prediction[0]),
            'probability_yes': float(probability[0][1]),
            'probability_no': float(probability[0][0]),
            'features': input_data.to_dict(orient='records')[0]
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if metrics is None:
        return jsonify({'error': 'Metrics not available'}), 500
    return json.dumps(metrics, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        importances = model.feature_importances_
        features = [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
            'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
            'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
            'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
        ]
        return jsonify({
            'features': features,
            'importances': importances.tolist()
        })
    except Exception as e:
        logging.error(f"Error in feature_importance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'api_version': '1.0'
    })

if __name__ == '__main__':
    app.run(debug=True)
