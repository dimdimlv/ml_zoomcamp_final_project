"""
Simple test to verify pipelines work correctly
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Testing Training Pipeline")
print("="*60)

try:
    from src.pipeline.train_pipeline import TrainPipeline
    
    print("\n✓ Successfully imported TrainPipeline")
    print("\nStarting training (this may take a few minutes)...")
    
    # Create and run the training pipeline
    train_pipeline = TrainPipeline(use_hyperparameter_tuning=False)
    results = train_pipeline.start_training()
    
    print("\n" + "="*60)
    print("Training Results:")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Model Path: {results['model_path']}")
    print(f"Preprocessor Path: {results['preprocessor_path']}")
    print("\n✅ Training pipeline test PASSED")
    
except Exception as e:
    print(f"\n❌ Training pipeline test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Testing Prediction Pipeline")
print("="*60)

try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomData
    import pandas as pd
    
    print("\n✓ Successfully imported PredictPipeline and CustomData")
    
    # Create sample data
    sample_data = CustomData(
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
    
    print("\n✓ Created sample data")
    
    # Convert to DataFrame
    df = sample_data.get_data_as_dataframe()
    print(f"\n✓ Converted to DataFrame with shape: {df.shape}")
    
    # Make prediction
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(df)
    
    print("\n" + "="*60)
    print("Prediction Results:")
    print("="*60)
    print(f"Input: Male, Age 25, Height 1.75m, Weight 70.5kg")
    print(f"Predicted Obesity Level: {predictions[0]}")
    print("\n✅ Prediction pipeline test PASSED")
    
except Exception as e:
    print(f"\n❌ Prediction pipeline test FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 All Pipeline Tests PASSED!")
print("="*60)
print("\nYou can now run the Flask application:")
print("  python app.py")
print("\nOr test the API endpoints:")
print("  python test_app.py")
print("="*60)
