import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """Pipeline for training the ML model."""
    
    def __init__(self, use_hyperparameter_tuning: bool = False):
        """
        Initialize the training pipeline.
        
        Args:
            use_hyperparameter_tuning (bool): Whether to use hyperparameter tuning.
        """
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        
    def start_training(self):
        """
        Execute the complete training pipeline.
        
        Returns:
            dict: Dictionary containing training results and metrics.
        """
        try:
            logging.info("="*50)
            logging.info("Starting training pipeline")
            logging.info("="*50)
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation(DataTransformationConfig())
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer(use_tuning=self.use_hyperparameter_tuning)
            accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info("="*50)
            logging.info(f"Training pipeline completed successfully!")
            logging.info(f"Model accuracy: {accuracy:.4f}")
            logging.info("="*50)
            
            return {
                "status": "success",
                "accuracy": accuracy,
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "preprocessor_path": preprocessor_path,
                "model_path": model_trainer.model_trainer_config.trained_model_file_path
            }
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run the training pipeline
    pipeline = TrainPipeline(use_hyperparameter_tuning=False)
    results = pipeline.start_training()
    print(f"\nTraining completed with accuracy: {results['accuracy']:.4f}")
