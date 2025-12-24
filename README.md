# Obesity Level Prediction

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

## Project Structure

```
ml_zoomcamp_final_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup configuration
├── notebooks/
│   ├── 0-obesity_dataset_demo.ipynb     # Dataset exploration
│   └── 1-EDA_obesity_dataset.ipynb      # Exploratory Data Analysis
├── src/
│   ├── components/
│   │   ├── data_ingestion.py            # Load and prepare data
│   │   ├── data_transformation.py       # Feature engineering & preprocessing
│   │   └── model_trainer.py             # Model training & evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py            # Full training pipeline
│   │   └── predict_pipeline.py          # Inference pipeline
│   ├── utils.py                         # Utility functions
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
python src/pipeline/train_pipeline.py
```

This will:
1. Load and preprocess the data
2. Train multiple models (Logistic Regression, Random Forest, Gradient Boosting, etc.)
3. Evaluate models using cross-validation
4. Save the best model to a pickle file

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
    "NCP": 3.0,
    "CAEC": "Between_meals",
    "SMOKE": "no",
    "CH2O": 2.5,
    "SCC": "no",
    "FAF": 3.0,
    "TUE": 0.0,
    "CALC": "Sometimes",
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
3. Execute the training pipeline: `python src/pipeline/train_pipeline.py`
4. Test the prediction service: `python src/pipeline/predict_pipeline.py`

## Dependencies

Key packages used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and preprocessing
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
