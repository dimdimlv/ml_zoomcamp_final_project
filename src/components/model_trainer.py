import os
import sys
from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import save_object, load_hyperparameters, get_model_params
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    hyperparams_config_path: str = "params.yaml"
    use_hyperparameter_tuning: bool = False  # Set to False by default to avoid long waits
    
class ModelTrainer:
    def __init__(self, use_tuning: bool = False):
        self.model_trainer_config = ModelTrainerConfig(use_hyperparameter_tuning=use_tuning)
        self.hyperparams = load_hyperparameters(self.model_trainer_config.hyperparams_config_path)
        
    def initiate_model_trainer(self, train_array, test_array):
        """Trains multiple models with hyperparameter tuning and saves the best one.

        Args:
            train_array: Training data with features and target.
            test_array: Testing data with features and target.

        Returns:
            The accuracy score of the best model.
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
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=n_classes,
                    tree_method="hist",
                ),
                "LightGBM": LGBMClassifier(objective="multiclass", num_class=n_classes, verbose=-1),
            }
            
            # Perform hyperparameter tuning if enabled
            if self.model_trainer_config.use_hyperparameter_tuning:
                logging.info("Hyperparameter tuning enabled")
                tuned_models = self._tune_hyperparameters(X_train, y_train, models)
            else:
                logging.info("Using default model parameters")
                tuned_models = models
            
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                models=tuned_models
            )
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))
            
            # Get the best model name from the report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = tuned_models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            
            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score:.4f}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            logging.info(f"Best model ({best_model_name}) accuracy score: {acc:.4f}")

            return acc
            
        except Exception as e:
            raise CustomException(e, sys)

    def _tune_hyperparameters(self, X_train, y_train, models):
        """Performs GridSearchCV hyperparameter tuning for all models.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            models: Dictionary of models to tune.
            
        Returns:
            Dictionary of tuned models with best parameters.
        """
        try:
            tuned_models = {}
            cv_splits = self.hyperparams.get("cv_splits", 5)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            
            for model_name, model in models.items():
                logging.info(f"Tuning hyperparameters for {model_name}...")
                
                # Get hyperparameters for this model
                param_grid = get_model_params(self.hyperparams, model_name)
                
                if not param_grid:
                    logging.warning(f"No hyperparameters found for {model_name}, using default")
                    tuned_models[model_name] = model
                    continue
                
                # Remove non-grid parameters
                non_grid_params = ["random_state", "tree_method"]
                grid_params = {k: v for k, v in param_grid.items() 
                              if k not in non_grid_params}
                
                if not grid_params:
                    logging.warning(f"No grid parameters for {model_name}, using default")
                    tuned_models[model_name] = model
                    continue
                
                # Perform GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=grid_params,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Update model with best parameters
                best_params = grid_search.best_params_
                logging.info(f"{model_name} best parameters: {best_params}")
                logging.info(f"{model_name} best CV accuracy: {grid_search.best_score_:.4f}")
                
                tuned_models[model_name] = grid_search.best_estimator_
            
            return tuned_models
            
        except Exception as e:
            logging.error(f"Error occurred during hyperparameter tuning: {e}")
            raise CustomException(e, sys)