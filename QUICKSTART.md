# Quick Start Guide - Flask Obesity Prediction App

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python test_pipelines.py
```
This will:
- Download the dataset
- Train the model
- Test predictions
- Create all necessary artifacts

### Step 3: Run the Flask App
```bash
python app.py
```
Then open your browser to: `http://localhost:5000`

---

## 📋 What You Can Do

### Web Interface
1. **Home Page** (`/`) - Start here
2. **Make Predictions** (`/predictdata`) - Predict obesity levels
3. **Train Model** (`/train`) - Retrain with latest data

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Make a Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Height": 1.75,
    "Weight": 70.5,
    "FCVC": 2.5,
    "NCP": 3,
    "CH2O": 2,
    "FAF": 1.5,
    "TUE": 1,
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

#### Train the Model
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"use_hyperparameter_tuning": false}'
```

---

## 🧪 Testing

### Test Pipelines
```bash
python test_pipelines.py
```

### Test API (requires Flask app running)
```bash
# Terminal 1
python app.py

# Terminal 2
python test_app.py
```

---

## 📁 Project Structure

```
├── app.py                    # Flask application
├── test_pipelines.py         # Pipeline tests
├── test_app.py               # API tests
├── template/                 # HTML templates
│   ├── index.html           # Home page
│   ├── home.html            # Prediction form
│   └── train.html           # Training page
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training logic
│   │   └── predict_pipeline.py  # Prediction logic
│   └── components/          # ML components
└── artifacts/               # Generated models & data
```

---

## 💡 Tips

- First time? Run `test_pipelines.py` before starting the Flask app
- Training without hyperparameter tuning takes ~2-5 minutes
- With hyperparameter tuning enabled, expect 10-30 minutes
- Check `logs/` directory for detailed execution logs
- Model and preprocessor are saved in `artifacts/` directory

---

## ❓ Common Issues

**Problem**: "Model not found" error  
**Solution**: Run `python test_pipelines.py` first

**Problem**: Port 5000 already in use  
**Solution**: Edit `app.py` and change port to 5001

**Problem**: Import errors  
**Solution**: Make sure virtual environment is activated and dependencies installed

---

## 📖 Full Documentation

See [README.md](README.md) for complete documentation including:
- Detailed API reference
- Feature descriptions
- Model information
- Advanced usage examples
