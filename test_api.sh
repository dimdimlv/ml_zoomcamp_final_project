#!/bin/bash
# API Test Commands for Obesity Prediction Flask App
# Make sure the Flask app is running: python app.py

echo "=================================================="
echo "Obesity Prediction API - Test Commands"
echo "=================================================="
echo ""

# Health Check
echo "1. Health Check"
echo "Command:"
echo 'curl http://localhost:5000/health'
echo ""
echo "Running..."
curl http://localhost:5000/health
echo ""
echo ""

# Prediction - Normal Weight Example
echo "=================================================="
echo "2. Prediction - Normal Weight Example"
echo "Command:"
cat << 'EOF'
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
EOF
echo ""
echo "Running..."
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
echo ""
echo ""

# Prediction - Potential Obesity Example
echo "=================================================="
echo "3. Prediction - Potential Obesity Example"
echo "Command:"
cat << 'EOF'
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35.0,
    "Height": 1.68,
    "Weight": 95.0,
    "FCVC": 1.0,
    "NCP": 4.0,
    "CH2O": 1.0,
    "FAF": 0.0,
    "TUE": 2.0,
    "Gender": "Female",
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "CAEC": "Always",
    "SMOKE": "no",
    "SCC": "no",
    "CALC": "Frequently",
    "MTRANS": "Automobile"
  }'
EOF
echo ""
echo "Running..."
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35.0,
    "Height": 1.68,
    "Weight": 95.0,
    "FCVC": 1.0,
    "NCP": 4.0,
    "CH2O": 1.0,
    "FAF": 0.0,
    "TUE": 2.0,
    "Gender": "Female",
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "CAEC": "Always",
    "SMOKE": "no",
    "SCC": "no",
    "CALC": "Frequently",
    "MTRANS": "Automobile"
  }'
echo ""
echo ""

# Train Model (commented out as it takes time)
echo "=================================================="
echo "4. Train Model (Not executed - takes several minutes)"
echo "Command:"
cat << 'EOF'
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "use_hyperparameter_tuning": false
  }'
EOF
echo ""
echo "To run training, uncomment the section below and execute"
echo ""

# Uncomment to actually run training:
# curl -X POST http://localhost:5000/api/train \
#   -H "Content-Type: application/json" \
#   -d '{
#     "use_hyperparameter_tuning": false
#   }'

echo "=================================================="
echo "Test Commands Completed"
echo "=================================================="
