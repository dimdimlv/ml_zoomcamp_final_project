# Obesity Level Prediction - ML Zoomcamp Final Project

## Problem Description

This project aims to predict obesity levels in individuals based on their eating habits and physical condition. Obesity is a major public health concern associated with numerous chronic diseases including diabetes, cardiovascular disease, and certain cancers. 

The goal is to build a machine learning model that can:
- Classify individuals into different obesity categories
- Identify key factors contributing to obesity
- Enable early intervention through risk assessment

This predictive model can be valuable for:
- Healthcare providers for patient risk stratification
- Fitness and wellness applications
- Public health initiatives for obesity prevention

## Dataset

**Source**: Estimation of Obesity Levels Based On Eating Habits and Physical Condition [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z.

The dataset contains information about:
- **Eating habits**: Frequency of consumption of high caloric food, vegetables, water intake, etc.
- **Physical condition**: Physical activity frequency, time spent using technology, etc.
- **Demographic factors**: Age, sex, height, weight
- **Target variable**: Obesity level (7 classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III)

## Flask Web Application

This project includes a complete Flask web application with both training and prediction pipelines.

### Features

- **🏠 Home Page**: Beautiful landing page with navigation
- **📊 Prediction Interface**: Web form for making obesity level predictions
- **🎯 Training Interface**: Web interface to train the model with latest data
- **🔌 REST API Endpoints**: JSON API for programmatic access
- **✅ Health Check**: Monitor application status

## Project Structure

```
ml_zoomcamp_final_project/
├── README.md                 # This file
├── app.py                    # Flask application with routes
├── requirements.txt          # Python dependencies
├── params.yaml               # Hyperparameter configurations
├── setup.py                  # Package setup configuration
├── test_pipelines.py         # Test script for pipelines
├── test_app.py               # Test script for API endpoints
├── artifacts/                # Generated files (data, models)
│   ├── data.csv              # Raw dataset
│   ├── train.csv             # Training set
│   ├── test.csv              # Test set
│   ├── model.pkl             # Trained model
│   └── preprocessor.pkl      # Fitted preprocessor
├── logs/                     # Training logs
├── template/                 # HTML templates
│   ├── index.html            # Home page
│   ├── home.html             # Prediction form
│   └── train.html            # Training page
├── notebooks/
│   ├── 0-obesity_dataset_demo.ipynb     # Dataset exploration
│   └── 1-EDA_obesity_dataset.ipynb      # Exploratory Data Analysis
├── src/
│   ├── components/
│   │   ├── data_ingestion.py            # Load and prepare data
│   │   ├── data_transformation.py       # Feature engineering & preprocessing
│   │   └── model_trainer.py             # Model training & hyperparameter tuning
│   ├── pipeline/
│   │   ├── train_pipeline.py            # Training pipeline
│   │   └── predict_pipeline.py          # Prediction pipeline
│   ├── utils.py              # Utility functions
│   ├── logger.py             # Logging configuration
│   └── exception.py          # Custom exception handling
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ml_zoomcamp_final_project
```

2. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Test the pipelines** (recommended first step):
```bash
python test_pipelines.py
```
This will:
- Train the model with default settings
- Test the prediction pipeline
- Verify everything works correctly

2. **Run the Flask application**:
```bash
python app.py
```
The application will start on `http://localhost:5000`

3. **Access the web interface**:
- Open your browser and navigate to `http://localhost:5000`
- Use the web forms to train the model or make predictions

### Web Interface Routes

#### 1. Home Page
- **URL**: `http://localhost:5000/`
- **Description**: Landing page with navigation to training and prediction

#### 2. Prediction Form
- **URL**: `http://localhost:5000/predictdata`
- **Method**: GET (display form), POST (submit prediction)
- **Description**: Interactive form to input features and get obesity level prediction

**Required Input Fields**:
- Age (years)
- Height (meters)
- Weight (kilograms)
- Gender (Male/Female)
- Family history of overweight (yes/no)
- Frequent high caloric food consumption (yes/no)
- Vegetable consumption frequency (0-3)
- Number of main meals (1-4)
- Food consumption between meals (no/Sometimes/Frequently/Always)
- Smoking (yes/no)
- Daily water intake (0-3 liters)
- Calorie consumption monitoring (yes/no)
- Physical activity frequency (0-3)
- Technology use time (0-2 hours)
- Alcohol consumption (no/Sometimes/Frequently/Always)
- Transportation mode (Public_Transportation/Automobile/Walking/Motorbike/Bike)

#### 3. Training Interface
- **URL**: `http://localhost:5000/train`
- **Method**: GET (display form), POST (start training)
- **Description**: Interface to train/retrain the model
- **Options**: Enable hyperparameter tuning (checkbox)

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

#### 2. Predict (API)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25.0,
    "Height": 1.75,
    "Weight": 70.5,
    "FCVC": 2.5,
    "NCP": 3.0,
    "CH2O": 2.0,
    "FAF": 1.5,
    "TUE": 1.0,
    "Gender": "Male",
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "SCC": "no",
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
  }'
```

**Response**:
```json
{
  "status": "success",
  "prediction": "Normal_Weight"
}
```

#### 3. Train (API)
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "use_hyperparameter_tuning": false
  }'
```

**Response**:
```json
{
  "status": "success",
  "accuracy": 0.9234,
  "model_path": "artifacts/model.pkl"
}
```

### Testing

#### Test Pipelines
```bash
python test_pipelines.py
```
This tests:
- Training pipeline functionality
- Prediction pipeline functionality
- Data flow through the system

#### Test API Endpoints
```bash
# Make sure Flask app is running first
python app.py

# In another terminal
python test_app.py
```
This tests:
- Health check endpoint
- Training API endpoint
- Prediction API endpoint

### Using Python Scripts Directly

#### Training Pipeline
```python
from src.pipeline.train_pipeline import TrainPipeline

# Train without hyperparameter tuning (faster)
pipeline = TrainPipeline(use_hyperparameter_tuning=False)
results = pipeline.start_training()
print(f"Model accuracy: {results['accuracy']:.4f}")

# Train with hyperparameter tuning (more accurate, slower)
pipeline = TrainPipeline(use_hyperparameter_tuning=True)
results = pipeline.start_training()
```

#### Prediction Pipeline
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create sample data
data = CustomData(
    Age=25.0,
    Height=1.75,
    Weight=70.5,
    FCVC=2.5,
    NCP=3.0,
    CH2O=2.0,
    FAF=1.5,
    TUE=1.0,
    Gender="Male",
    family_history_with_overweight="yes",
    FAVC="yes",
    CAEC="Sometimes",
    SMOKE="no",
    SCC="no",
    CALC="Sometimes",
    MTRANS="Public_Transportation"
)

# Make prediction
df = data.get_data_as_dataframe()
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)
print(f"Predicted obesity level: {prediction[0]}")
```

## Models and Performance

The project trains and evaluates multiple classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The best performing model is automatically selected and saved based on test accuracy.

## Logging

All operations are logged to the `logs/` directory with timestamps. Logs include:
- Data ingestion progress
- Transformation steps
- Model training metrics
- Prediction requests
- Error messages and stack traces

## Troubleshooting

### Model Not Found Error
If you get an error about missing model files:
```bash
python test_pipelines.py
```
This will train the model and create all necessary artifacts.

### Port Already in Use
If port 5000 is already in use:
```python
# Edit app.py and change the port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Changed to 5001
```

### Module Import Errors
Ensure you're in the correct directory and virtual environment:
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Future Improvements

- Add user authentication and session management
- Implement prediction history tracking
- Add data visualization dashboards
- Deploy to cloud platforms (AWS, Azure, GCP)
- Add Docker containerization
- Implement A/B testing for model versions
- Add batch prediction capabilities

## License

This project is part of the ML Zoomcamp course final project.

## Acknowledgments

- Dataset from UCI Machine Learning Repository
- ML Zoomcamp course and community
- Flask framework documentation

│   │   ├── train_pipeline.py            # Full training pipeline
│   │   └── predict_pipeline.py          # Inference pipeline
│   ├── utils.py                         # Utility functions (incl. YAML loading)
│   ├── logger.py                        # Logging configuration
│   └── exception.py                     # Custom exceptions
└── docs/
    ├── Project_README.md                # ML Zoomcamp project guidelines
    └── project-tips.md                  # Best practices checklist
```

## Installation

### Prerequisites
- Python 3.13+
- pip or conda

### Setup Instructions

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd ml_zoomcamp_final_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation & EDA

Explore the data using the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/0-obesity_dataset_demo.ipynb    # Dataset overview
jupyter notebook notebooks/1-EDA_obesity_dataset.ipynb     # In-depth analysis
```

These notebooks include:
- Data loading and structure exploration
- Missing value analysis
- Distribution of features and target variable
- Correlation analysis
- Feature importance identification

## Model Training

Train the final model using the training pipeline:

```bash
# Run the complete data ingestion + transformation + training pipeline
python -m src.components.data_ingestion
```

This will:
1. **Data Ingestion**: Fetch dataset from UCI repository and split into train/test sets
2. **Data Transformation**: Preprocess features using StandardScaler and OneHotEncoder
3. **Model Training**: Train and evaluate 6 different models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
4. **Model Selection**: Select best model based on test accuracy
5. **Save Artifacts**: Save trained model and preprocessor to `artifacts/`

### Hyperparameter Tuning

The project supports **optional hyperparameter tuning** with GridSearchCV:

**Default Mode (Fast)**: Uses default model parameters
```bash
python -m src.components.data_ingestion
```

**With Hyperparameter Tuning**: Enables grid search for optimal parameters
```python
# In data_ingestion.py or custom script:
from src.components.model_trainer import ModelTrainer

modeltrainer = ModelTrainer(use_tuning=True)  # Enable tuning
accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr)
```

Hyperparameters are configured in `params.yaml` with search grids for each model:
- **Logistic Regression**: `max_iter`, `C` (regularization)
- **Decision Tree**: `max_depth`, `min_samples_split`
- **Random Forest**: `n_estimators`, `max_depth`
- **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`
- **XGBoost**: `n_estimators`, `learning_rate`, `max_depth`
- **LightGBM**: `n_estimators`, `learning_rate`, `num_leaves`

You can easily customize hyperparameter ranges by editing `params.yaml`.

## Model Prediction

Make predictions using the trained model:

```bash
python src/pipeline/predict_pipeline.py
```

The prediction pipeline loads the trained model and can make inferences on new data.

## Web Service API

The model is served via a Flask web service for easy integration.

### Running the Service Locally

```bash
python src/pipeline/predict_pipeline.py  # or serve.py if available
```

The service will be available at `http://localhost:9696`

### API Endpoints

**Health Check**:
```bash
curl http://localhost:9696/health
```

**Make Prediction**:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 25.5,
    "Height": 1.75,
    "Weight": 75.0,
    "FCVC": 2.0,
    model training pipeline evaluates all 6 models using:
- **Stratified K-Fold Cross-Validation** (3 or 5 folds, configurable in `params.yaml`)
- **Holdout test set** for final evaluation

**Current Best Model Performance**:
- **Test Accuracy**: **97.16%**
- **Primary Metric**: Accuracy score
- **Secondary Metric**: F1-macro score (logged during CV)

All models are compared using:
- Cross-validation accuracy and F1-macro scores
- Final holdout test accuracy

The best performing model is automatically selected and saved to `artifacts/model.pkl`.

See the notebooks and training logetimes",
    "MTRANS": "Public_Transportation"
  }'
```

**Expected Response**:
```json
{
  "prediction": "Normal_Weight",
  "confidence": 0.92
}
```

## Docker Deployment

Build and run the service in a Docker container:

```bash
# Build the image
docker build -t obesity-prediction .

# Run the container
docker run -it -p 9696:9696 obesity-prediction
```

The service will be available at `http://localhost:9696`

## Model Performance

The final model achieves the following metrics on the test set:

- **Accuracy**: [To be filled after training]
- **Precision**: [To be filled after training]
- **Recall**: [To be filled after training]
- **F1-Score**: [To be filled after training]

See the notebooks for detailed performance comparisons between different models.

## Key Features Used

Based on EDA and feature importance analysis:
1. **Physical characteristics**: Height, Weight
2. **Eating habits**: Caloric food frequency (FCVC), Water consumption (CH2O), Caloric beverages (CALC)
3. **Physical activity**: Frequency of physical activity (FAF), Time using technology (TUE)
4. **Lifestyle factors**: Smoking status, Mode of transportation

## Reproducing Results

To reproduce the complete analysis and model training:

1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebooks in order: `0-obesity_dataset_demo.ipynb` → `1-EDA_obesity_dataset.ipynb`
3. Execute the training pipeline: `python s, preprocessing, and hyperparameter tuning
- **xgboost**: Extreme Gradient Boosting classifier
- **lightgbm**: Light Gradient Boosting classifier
- **matplotlib/seaborn**: Data visualization
- **ucimlrepo**: Downloading datasets from UCI ML Repository
- **pyyaml**: Loading hyperparameter configurations
- *Implemented Features

- [x] **Hyperparameter tuning** with GridSearchCV (configurable via `params.yaml`)
- [x] **Multiple model comparison** (6 algorithms)
- [x] **YAML-based configuration** for easy hyperparameter management
- [x] **Stratified K-Fold cross-validation** for robust evaluation
- [x] **Automated model selection** based on accuracy
- [x] **Comprehensive logging** system
- [x] **Modular pipeline architecture** (ingestion → transformation → training)
- [x] **Flexible tuning mode** (on/off for fast prototyping)

## Future Improvements

- [ ] Add more advanced feature engineering
- [ ] Deploy to cloud platform (AWS, GCP, or Azure)
- [ ] Create a web UI for easier interaction
- [ ] Add model explainability with SHAP values
- [ ] Implement continuous model retraining pipeline
- [ ] Add RandomizedSearchCV option for larger parameter spaces
- [ ] Implement ensemble methods (stacking, voting)
- [ ] Add automated hyperparameter optimization (Optuna, Hyperopt)rocessing
- **matplotlib**: Data visualization
- **ucimlrepo**: Downloading datasets from UCI ML Repository

See `requirements.txt` for complete dependency list.

## Future Improvements

- [ ] Implement hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Add more advanced feature engineering
- [ ] Deploy to cloud platform (AWS, GCP, or Azure)
- [ ] Create a web UI for easier interaction
- [ ] Add model explainability with SHAP values
- [ ] Implement continuous model retraining pipeline

## References

1. [ML Zoomcamp Obesity Dataset](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-body-index)
2. [UCI ML Repository - Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
3. Palechor, F. A., & Manotas, A. D. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in brief, 25, 104344.

## License

This project is part of the ML Zoomcamp coursework.

## Contact

**Author**: Dmitry Polischuk  
**Email**: dmitry.polischuk@gmail.com
