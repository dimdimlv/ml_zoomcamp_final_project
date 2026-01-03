"""
Test script for Flask application endpoints
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*50)
    print("Testing Health Check Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_api_predict():
    """Test the API prediction endpoint"""
    print("\n" + "="*50)
    print("Testing API Prediction Endpoint")
    print("="*50)
    
    # Sample data for prediction
    sample_data = {
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
    }
    
    print("Sample Input:")
    print(json.dumps(sample_data, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=sample_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_api_train():
    """Test the API training endpoint"""
    print("\n" + "="*50)
    print("Testing API Training Endpoint")
    print("="*50)
    print("⚠️  This may take several minutes...")
    
    train_data = {
        "use_hyperparameter_tuning": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/train",
        json=train_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("Starting Flask Application Tests")
    print("🧪 "*20)
    
    results = {
        "health_check": False,
        "api_train": False,
        "api_predict": False
    }
    
    # Test 1: Health Check
    try:
        results["health_check"] = test_health_check()
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
    
    # Test 2: Training (run first to ensure model exists)
    try:
        results["api_train"] = test_api_train()
    except Exception as e:
        print(f"❌ API training failed: {str(e)}")
    
    # Test 3: Prediction
    try:
        results["api_predict"] = test_api_predict()
    except Exception as e:
        print(f"❌ API prediction failed: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*50)
    all_passed = all(results.values())
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("="*50)

if __name__ == "__main__":
    print("\n⚠️  Make sure the Flask application is running on http://localhost:5000")
    print("Run: python app.py")
    input("\nPress Enter to start tests...")
    run_all_tests()
