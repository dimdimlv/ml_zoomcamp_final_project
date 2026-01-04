# Obesity Level Prediction - ML Zoomcamp Final Project

## 📋 Problem Description

Obesity is a critical global health challenge affecting millions of people worldwide. According to the World Health Organization, worldwide obesity has nearly tripled since 1975, with significant implications for healthcare systems and individual well-being.

### The Problem

This project addresses the need for **early identification and classification of obesity risk levels** in individuals based on their lifestyle patterns, eating habits, and physical characteristics.

### The Solution

A machine learning classification system that:
- **Predicts obesity levels** across 7 categories from underweight to severe obesity
- **Identifies key risk factors** through feature importance analysis
- **Provides instant risk assessment** via a web service API
- **Enables early intervention** by flagging individuals at risk

### Real-World Applications

1. **Healthcare Providers**: Quick risk stratification during patient screenings
2. **Wellness Apps**: Integration into fitness and nutrition applications
3. **Public Health**: Population-level obesity monitoring and prevention programs
4. **Research**: Understanding correlations between lifestyle factors and obesity

### How It's Used

Users can input demographic data, eating habits, and physical activity patterns through either:
- A **web interface** for individual assessments
- A **REST API** for integration into existing healthcare systems

The model returns the predicted obesity category, enabling targeted interventions and personalized health recommendations.

## 📊 Dataset

**Source**: [Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://doi.org/10.24432/C5H31Z), UCI Machine Learning Repository (2019)

**Dataset Size**: 2,111 samples with 17 features

**Features Include**:
- **Physical Characteristics**: Age, Height, Weight, Gender
- **Eating Habits**: Vegetable consumption frequency (FCVC), Number of main meals (NCP), Food between meals (CAEC), High caloric food consumption (FAVC), Water intake (CH2O), Alcohol consumption (CALC)
- **Physical Activity**: Physical activity frequency (FAF), Time using technology devices (TUE)
- **Other Factors**: Family history of overweight, Smoking habits (SMOKE), Calorie monitoring (SCC), Transportation mode (MTRANS)

**Target Variable**: Obesity level with 7 classes:
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

## 🎯 Project Features

- ✅ **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
- ✅ **Hyperparameter Tuning**: Optional GridSearchCV with configurable parameters via YAML
- ✅ **Complete Web Service**: Flask application with REST API endpoints
- ✅ **Docker Support**: Fully containerized application ready for deployment
- ✅ **Comprehensive Testing**: Unit tests for pipelines and API endpoints
- ✅ **Logging System**: Detailed logging for debugging and monitoring
- ✅ **Modular Architecture**: Clean separation of concerns with pipeline design pattern

## 📁 Project Structure

```
ml_zoomcamp_final_project/
├── README.md                     # Project documentation (this file)
├── DOCKER.md                     # Docker deployment guide
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml            # Docker Compose orchestration
├── .dockerignore                 # Docker build exclusions
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
├── params.yaml                   # Model hyperparameters configuration
├── setup.py                      # Package installation setup
├── test_pipelines.py             # Pipeline integration tests
├── test_app.py                   # API endpoint tests
├── test_api.sh                   # Bash script for API testing
├── artifacts/                    # Generated artifacts (models, data)
│   ├── data.csv                  # Full dataset
│   ├── train.csv                 # Training split
│   ├── test.csv                  # Test split
│   ├── model.pkl                 # Trained model (after training)
│   └── preprocessor.pkl          # Fitted preprocessor (after training)
├── logs/                         # Application logs
├── template/                     # HTML templates for web UI
│   ├── index.html                # Landing page
│   ├── home.html                 # Prediction form
│   └── train.html                # Training interface
├── notebooks/                    # Jupyter notebooks for EDA
│   ├── 0-obesity_dataset_demo.ipynb        # Dataset overview
│   └── 1-EDA_obesity_dataset.ipynb         # Exploratory analysis
└── src/                          # Source code modules
    ├── __init__.py
    ├── exception.py              # Custom exception handling
    ├── logger.py                 # Logging configuration
    ├── utils.py                  # Utility functions (YAML, save/load)
    ├── components/               # Pipeline components
    │   ├── __init__.py
    │   ├── data_ingestion.py     # Data loading and splitting
    │   ├── data_transformation.py # Feature engineering
    │   └── model_trainer.py      # Model training and selection
    └── pipeline/                 # End-to-end pipelines
        ├── __init__.py
        ├── train_pipeline.py     # Complete training workflow
        └── predict_pipeline.py   # Inference workflow
```

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- Docker (optional, for containerized deployment)

### Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/dimdimlv/ml_zoomcamp_final_project.git
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

4. **Run initial training** (creates model artifacts):
```bash
python test_pipelines.py
```
This will train the model and create `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

5. **Start the Flask application**:
```bash
python app.py
```
Access the web interface at `http://localhost:5000`

### Docker Deployment (Recommended)

**Quick start with Docker Compose**:
```bash
docker-compose up --build
```

**Or using Docker directly**:
```bash
# Build the image
docker build -t obesity-ml-app .

# Run the container
docker run -d -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts obesity-ml-app
```

See [DOCKER.md](DOCKER.md) for detailed deployment instructions.

## 📓 Data Exploration & EDA

Comprehensive exploratory data analysis is available in Jupyter notebooks:

```bash
jupyter notebook notebooks/0-obesity_dataset_demo.ipynb
jupyter notebook notebooks/1-EDA_obesity_dataset.ipynb
```

**Key EDA Findings**:
- **Class Distribution**: Balanced distribution across 7 obesity categories
- **Missing Values**: No missing data in the dataset
- **Feature Correlations**: Strong correlations between Weight, Height, and obesity levels
- **Important Features**: Weight, Height, Family history, Physical activity frequency
- **Data Quality**: High-quality synthetic dataset with realistic patterns

The notebooks include:
- Data structure and statistics
- Distribution plots for numerical features
- Count plots for categorical features
- Correlation heatmaps
- Feature importance analysis
- Target variable distribution

## 🤖 Model Training

### Training Pipeline

The project trains and compares 6 different classification algorithms:

1. **Logistic Regression** (baseline)
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **XGBoost**
6. **LightGBM**

### Running Training

**Option 1: Using the test script (recommended for first run)**:
```bash
python test_pipelines.py
```

**Option 2: Using the training pipeline directly**:
```python
from src.pipeline.train_pipeline import TrainPipeline

# Train without hyperparameter tuning (fast, ~2-3 minutes)
pipeline = TrainPipeline(use_hyperparameter_tuning=False)
results = pipeline.start_training()
print(f"Model accuracy: {results['accuracy']:.4f}")

# Train with hyperparameter tuning (slower, ~10-15 minutes)
pipeline = TrainPipeline(use_hyperparameter_tuning=True)
results = pipeline.start_training()
```

**Option 3: Via web interface**:
1. Start the Flask app: `python app.py`
2. Navigate to `http://localhost:5000/train`
3. Click "Train Model" (optionally enable hyperparameter tuning)

**Option 4: Via API**:
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"use_hyperparameter_tuning": false}'
```

### Hyperparameter Tuning

Hyperparameters are configured in `params.yaml`. When tuning is enabled, GridSearchCV searches over parameter grids for each model:

**Example configuration**:
```yaml
models:
  LogisticRegression:
    max_iter: 1000
    random_state: 42
  RandomForest:
    n_estimators: 100
    random_state: 42
  XGBoost:
    learning_rate: 0.1
    n_estimators: 100
    random_state: 42

params:  # GridSearchCV parameter grids
  LogisticRegression:
    C: [0.01, 0.1, 1, 10]
    max_iter: [500, 1000]
  RandomForest:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, None]
```

### Model Selection & Performance

The training pipeline:
- Uses **stratified K-fold cross-validation** (5 folds)
- Evaluates models using **accuracy** and **F1-macro** scores
- Automatically selects the best model based on test accuracy
- Saves the best model to `artifacts/model.pkl`

**Expected Performance** (on test set):
- **Best Model**: Random Forest / XGBoost
- **Test Accuracy**: ~97% (varies by random seed)
- **Training Time**: 2-3 minutes (without tuning), 10-15 minutes (with tuning)

All training logs are saved to `logs/` directory with timestamps.

## 🌐 Web Service & API

The Flask application provides both a web UI and REST API endpoints.

### Web Interface Routes

#### 1. **Home Page** - `/`
Landing page with project overview and navigation links.

#### 2. **Prediction Form** - `/predictdata`
- **Methods**: GET (display form), POST (submit prediction)
- Interactive form to input patient data and get instant predictions
- Returns obesity classification with visual feedback

#### 3. **Training Interface** - `/train`
- **Methods**: GET (display interface), POST (start training)
- Trigger model retraining with optional hyperparameter tuning
- Displays training results and model accuracy

### REST API Endpoints

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

#### 2. Prediction API
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
    "FAF": 2.0,
    "TUE": 0.5,
    "Gender": "Male",
    "family_history_with_overweight": "no",
    "FAVC": "no",
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "SCC": "yes",
    "CALC": "no",
    "MTRANS": "Walking"
  }'
```

**Response**:
```json
{
  "status": "success",
  "prediction": "Normal_Weight"
}
```

**Input Fields Explained**:
- **Age**: Age in years (float)
- **Height**: Height in meters (float)
- **Weight**: Weight in kilograms (float)
- **FCVC**: Frequency of vegetable consumption (0-3, float)
- **NCP**: Number of main meals per day (1-4, float)
- **CH2O**: Daily water intake in liters (0-3, float)
- **FAF**: Physical activity frequency (0-3, float)
- **TUE**: Time using technology devices in hours (0-2, float)
- **Gender**: "Male" or "Female"
- **family_history_with_overweight**: "yes" or "no"
- **FAVC**: Frequent consumption of high caloric food - "yes" or "no"
- **CAEC**: Consumption of food between meals - "no", "Sometimes", "Frequently", "Always"
- **SMOKE**: Smoking habit - "yes" or "no"
- **SCC**: Calories consumption monitoring - "yes" or "no"
- **CALC**: Consumption of alcohol - "no", "Sometimes", "Frequently", "Always"
- **MTRANS**: Transportation used - "Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"

#### 3. Training API
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"use_hyperparameter_tuning": false}'
```

**Response**:
```json
{
  "status": "success",
  "accuracy": 0.9716,
  "model_path": "artifacts/model.pkl"
}
```

### Testing the API

Use the provided bash script to test all endpoints:
```bash
# Make sure the app is running first: python app.py
bash test_api.sh
```

Or use the Python test script:
```bash
python test_app.py
```

## 🐳 Docker Deployment

### Quick Start

**Using Docker Compose** (recommended):
```bash
docker-compose up --build
```

**Using Docker CLI**:
```bash
# Build the image
docker build -t obesity-ml-app .

# Run the container
docker run -d \
  --name obesity-ml-app \
  -p 5000:5000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/logs:/app/logs \
  obesity-ml-app
```

### Features

- ✅ **Persistent storage** for models and logs via volumes
- ✅ **Health monitoring** with built-in health checks
- ✅ **Auto-restart** on failure
- ✅ **Optimized builds** with layer caching
- ✅ **Clean image** with .dockerignore

### Testing the Containerized App

Once running, test the endpoints:
```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25, "Height": 1.75, "Weight": 70,
    "FCVC": 2, "NCP": 3, "CH2O": 2, "FAF": 2, "TUE": 1,
    "Gender": "Male", "family_history_with_overweight": "no",
    "FAVC": "no", "CAEC": "Sometimes", "SMOKE": "no",
    "SCC": "yes", "CALC": "no", "MTRANS": "Walking"
  }'
```

### Managing the Container

```bash
# View logs
docker logs -f obesity-ml-app

# Stop the container
docker-compose down

# Remove volumes (clears models and logs)
docker-compose down -v
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## 🧪 Testing & Validation

### Running Tests

**Test all pipelines**:
```bash
python test_pipelines.py
```
Tests: Data ingestion → Transformation → Training → Prediction

**Test API endpoints**:
```bash
# Terminal 1: Start the app
python app.py

# Terminal 2: Run tests
python test_app.py
```

**Test with bash script**:
```bash
bash test_api.sh
```

### Reproducibility

To reproduce the complete workflow from scratch:

1. **Clone the repository**:
```bash
git clone https://github.com/dimdimlv/ml_zoomcamp_final_project.git
cd ml_zoomcamp_final_project
```

2. **Set up environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run notebooks** (optional, for EDA):
```bash
jupyter notebook notebooks/0-obesity_dataset_demo.ipynb
jupyter notebook notebooks/1-EDA_obesity_dataset.ipynb
```

4. **Train the model**:
```bash
python test_pipelines.py
```

5. **Start the service**:
```bash
python app.py
```

6. **Or use Docker**:
```bash
docker-compose up --build
```

All steps are fully reproducible. The dataset is automatically downloaded from UCI ML Repository during the first run.

## 📊 Model Performance & Results

### Best Model Performance

**Current Results** (on holdout test set):
- **Best Model**: XGBoost / Random Forest (selected automatically)
- **Test Accuracy**: ~97%
- **Cross-Validation Accuracy**: ~96-97% (5-fold stratified)
- **F1-Macro Score**: ~96%

### Model Comparison

All 6 models are evaluated during training:

| Model | Test Accuracy | CV Accuracy | Training Time |
|-------|--------------|-------------|---------------|
| Logistic Regression | ~92% | ~91% | < 1 min |
| Decision Tree | ~94% | ~93% | < 1 min |
| Random Forest | ~97% | ~96% | 1-2 min |
| Gradient Boosting | ~96% | ~95% | 2-3 min |
| XGBoost | ~97% | ~96% | 2-3 min |
| LightGBM | ~96% | ~95% | 1-2 min |

*Note: Actual results may vary slightly due to random seed and data split*

### Feature Importance

Based on EDA and model analysis, the top contributing features are:

1. **Weight** - Most significant predictor
2. **Height** - Strong correlation with obesity levels
3. **Family history of overweight** - Genetic/environmental factors
4. **Physical activity frequency (FAF)** - Lifestyle indicator
5. **Age** - Metabolic changes with age
6. **Frequent high caloric food (FAVC)** - Diet quality indicator

### Evaluation Metrics

The model selection uses:
- **Primary Metric**: Accuracy (for balanced multi-class classification)
- **Secondary Metric**: F1-macro score (for class-level performance)
- **Validation Strategy**: Stratified K-Fold Cross-Validation (5 folds)
- **Final Test**: Holdout test set (20% of data)

## 💻 Technologies Used

### Core ML Stack
- **Python 3.8+**: Programming language
- **scikit-learn**: ML algorithms, preprocessing, and evaluation
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Web Service
- **Flask**: Web framework for API
- **requests**: HTTP client for testing

### Data Processing
- **StandardScaler**: Feature normalization
- **OneHotEncoder**: Categorical encoding
- **StratifiedKFold**: Cross-validation strategy

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

### Development Tools
- **Jupyter**: Interactive notebooks for EDA
- **pytest**: Unit testing
- **dill**: Advanced model serialization
- **PyYAML**: Configuration management
- **matplotlib/seaborn**: Data visualization

### Data Source
- **ucimlrepo**: UCI ML Repository Python client

## 🐛 Troubleshooting

### Model Not Found Error
```
FileNotFoundError: artifacts/model.pkl not found
```
**Solution**: Train the model first:
```bash
python test_pipelines.py
```

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change the port in `app.py`:
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Use different port
```

### Module Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Ensure you're in the project root and have activated the virtual environment:
```bash
cd ml_zoomcamp_final_project
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Docker Build Issues
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```
**Solution**: Ensure Docker has enough memory (at least 4GB) and try:
```bash
docker system prune -a
docker-compose build --no-cache
```

### Permission Issues with Volumes
**Solution**: Ensure the artifacts and logs directories have correct permissions:
```bash
chmod -R 755 artifacts logs
```

## 🚀 Future Improvements

### Short-term
- [ ] Add model explainability with SHAP values
- [ ] Implement batch prediction endpoint
- [ ] Add user authentication for training endpoint
- [ ] Create interactive dashboard with visualizations
- [ ] Add prediction confidence scores

### Long-term
- [ ] Deploy to cloud platform (AWS ECS, GCP Cloud Run, or Azure Container Apps)
- [ ] Implement A/B testing for model versions
- [ ] Add continuous model retraining pipeline
- [ ] Create mobile app interface
- [ ] Integrate with wearable devices for real-time data
- [ ] Add multi-language support

## 📚 References

1. **Dataset**: Palechor, F. M., & De La Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z
2. **ML Zoomcamp**: DataTalks.Club Machine Learning Zoomcamp - https://github.com/DataTalksClub/machine-learning-zoomcamp
3. **Flask Documentation**: https://flask.palletsprojects.com/
4. **scikit-learn Documentation**: https://scikit-learn.org/stable/
5. **XGBoost Documentation**: https://xgboost.readthedocs.io/
6. **Docker Documentation**: https://docs.docker.com/

## 📝 License

This project is created for educational purposes as part of the ML Zoomcamp course.

## 👤 Author

**Dmitry Polischuk**  
- Email: dmitry.polischuk@gmail.com
- GitHub: [@dimdimlv](https://github.com/dimdimlv)
- Project Repository: [ml_zoomcamp_final_project](https://github.com/dimdimlv/ml_zoomcamp_final_project)

## 🙏 Acknowledgments

- **DataTalks.Club** for the excellent ML Zoomcamp course
- **Alexey Grigorev** for course instruction and guidance
- **UCI Machine Learning Repository** for providing the dataset
- **ML Zoomcamp Community** for support and discussions
- **Open Source Community** for the amazing tools and libraries

---

**Note**: This is a final project for the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). For project evaluation criteria, see [docs/Project_README.md](docs/Project_README.md).
