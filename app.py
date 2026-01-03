from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Route for making predictions.
    GET: Display the prediction form
    POST: Process the form data and return prediction
    """
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Create CustomData object from form data
            data = CustomData(
                Age=float(request.form.get('Age')),
                Height=float(request.form.get('Height')),
                Weight=float(request.form.get('Weight')),
                FCVC=float(request.form.get('FCVC')),
                NCP=float(request.form.get('NCP')),
                CH2O=float(request.form.get('CH2O')),
                FAF=float(request.form.get('FAF')),
                TUE=float(request.form.get('TUE')),
                Gender=request.form.get('Gender'),
                family_history_with_overweight=request.form.get('family_history_with_overweight'),
                FAVC=request.form.get('FAVC'),
                CAEC=request.form.get('CAEC'),
                SMOKE=request.form.get('SMOKE'),
                SCC=request.form.get('SCC'),
                CALC=request.form.get('CALC'),
                MTRANS=request.form.get('MTRANS')
            )
            
            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            logging.info(f"Input data: {pred_df.to_dict()}")
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            logging.info(f"Prediction result: {results[0]}")
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            logging.error(f"Error in prediction route: {str(e)}")
            return render_template('home.html', results=f"Error: {str(e)}")

# Route for training the model
@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """
    Route for training the model.
    GET: Display the training page
    POST: Start the training process
    """
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            # Get hyperparameter tuning option from form (default: False)
            use_tuning = request.form.get('use_tuning', 'false').lower() == 'true'
            
            logging.info(f"Starting training with hyperparameter tuning: {use_tuning}")
            
            # Start training pipeline
            train_pipeline = TrainPipeline(use_hyperparameter_tuning=use_tuning)
            results = train_pipeline.start_training()
            
            message = f"Training completed successfully! Model accuracy: {results['accuracy']:.4f}"
            logging.info(message)
            
            return render_template('train.html', results=message, accuracy=results['accuracy'])
            
        except Exception as e:
            error_message = f"Error during training: {str(e)}"
            logging.error(error_message)
            return render_template('train.html', results=error_message)

# API endpoint for prediction (JSON)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for making predictions via JSON.
    """
    try:
        # Get JSON data from request
        json_data = request.get_json()
        
        # Create CustomData object
        data = CustomData(
            Age=float(json_data['Age']),
            Height=float(json_data['Height']),
            Weight=float(json_data['Weight']),
            FCVC=float(json_data['FCVC']),
            NCP=float(json_data['NCP']),
            CH2O=float(json_data['CH2O']),
            FAF=float(json_data['FAF']),
            TUE=float(json_data['TUE']),
            Gender=json_data['Gender'],
            family_history_with_overweight=json_data['family_history_with_overweight'],
            FAVC=json_data['FAVC'],
            CAEC=json_data['CAEC'],
            SMOKE=json_data['SMOKE'],
            SCC=json_data['SCC'],
            CALC=json_data['CALC'],
            MTRANS=json_data['MTRANS']
        )
        
        # Convert to DataFrame and make prediction
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return jsonify({
            'status': 'success',
            'prediction': str(results[0])
        })
        
    except Exception as e:
        logging.error(f"Error in API prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# API endpoint for training (JSON)
@app.route('/api/train', methods=['POST'])
def api_train():
    """
    API endpoint for training the model via JSON.
    """
    try:
        # Get JSON data from request
        json_data = request.get_json() or {}
        use_tuning = json_data.get('use_hyperparameter_tuning', False)
        
        # Start training
        train_pipeline = TrainPipeline(use_hyperparameter_tuning=use_tuning)
        results = train_pipeline.start_training()
        
        return jsonify({
            'status': 'success',
            'accuracy': results['accuracy'],
            'model_path': results['model_path']
        })
        
    except Exception as e:
        logging.error(f"Error in API training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the application is running.
    """
    model_exists = os.path.exists('artifacts/model.pkl')
    preprocessor_exists = os.path.exists('artifacts/preprocessor.pkl')
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_exists,
        'preprocessor_loaded': preprocessor_exists
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
