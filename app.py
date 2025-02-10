import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import sys
import tempfile
import mlflow
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.joblib')

sys.path.append(BASE_DIR)
from src.preprocess import preprocess_text
from src.mlflow_training import MLflowSpamTrainer
from src.model import SpamDetectionModel

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = SpamDetectionModel.load_model(MODEL_PATH, VECTORIZER_PATH)
        return model
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['email']
        
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})
        
        prediction = model.predict([email_text])[0]
        result = 'SPAM' if prediction == 1 else 'HAM'
        
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_model_params', methods=['GET'])
def get_model_params():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'No model found. Please train a model first.'})
            
        model = joblib.load(MODEL_PATH)
        params = model.get_params()
        metrics = {}
        if hasattr(model, 'best_score_'):
            metrics['best_score'] = model.best_score_
            
        return jsonify({
            'model_parameters': params,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving model parameters: {str(e)}'})

@app.route('/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            
            data = pd.read_csv(tmp_file.name)
            print(data.columns)
            # print("okay 1")
            # Verify required columns
            if 'sms' not in data.columns or 'label' not in data.columns:
                # print("okay 2")
                # os.unlink(tmp_file.name)
                # print("oaku 22")
                return jsonify({'error': 'CSV must contain "sms" and "label" columns'})
            # print("okay 3")
            # Preprocess text
            data['processed_text'] = data['sms'].apply(preprocess_text)
            # print(data)
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                data['processed_text'],
                data['label'],
                test_size=0.2,
                random_state=42
            )
            
            # Initialize trainer and optimize model
            trainer = MLflowSpamTrainer(
                experiment_name=f"spam_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            best_model, vectorizer = trainer.optimize_model(
                X_train, y_train, X_test, y_test
            )
            
            best_params = best_model.get_params()
            
            return jsonify({
                'message': 'Training completed successfully',
                'best_params': best_params,
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")

