import os
import sys
from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """Trains multiple models and saves the best one based on accuracy.

        Args:
            X: Features for training.
            y: Target variable for training.

        Returns:
            The path to the saved best model.
        """
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train_raw = train_array[:, :-1], train_array[:, -1]
            X_test, y_test_raw = test_array[:, :-1], test_array[:, -1]

            # Encode string labels to integers for all classifiers
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test = le.transform(y_test_raw)
            n_classes = len(le.classes_)
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=n_classes,
                    tree_method="hist",
                ),
                "LightGBM": LGBMClassifier(objective="multiclass", num_class=n_classes),
            }
            
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, 
                                                X_test = X_test, y_test = y_test, 
                                                models = models)
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))
            
            # Get the best model name from the report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing datasets")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            logging.info(f"Best model accuracy score: {acc}")

            return acc
            
        except Exception as e:
            raise CustomException(e, sys)