import os
import sys

import pandas as pd
import numpy as np
import yaml

from src.exception import CustomException
from src.logger import logging

import dill
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

def save_object(file_path: str, obj: object) -> None:
	"""Saves a Python object to a file using pickle.

	Args:
		file_path (str): The path where the object should be saved.
		obj (object): The Python object to be saved.

	Raises:
		CustomException: If there is an error during the saving process.
	"""
	try:
		dir_path = os.path.dirname(file_path)
		os.makedirs(dir_path, exist_ok=True)
		
		with open(file_path, 'wb') as file_obj:
			dill.dump(obj, file_obj)
			
		logging.info(f"Object saved successfully at {file_path}")
		
	except Exception as e:
		logging.error(f"Error occurred while saving object at {file_path}: {e}")
		raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
	"""Evaluates multiple models using stratified CV and test holdout.

	Returns a report mapping model name -> test_accuracy (primary metric).
	"""
	try:
		model_report = {}
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

		for name, model in models.items():
			# Stratified CV on train set
			cv_scores = cross_validate(
				model,
				X_train,
				y_train,
				cv=cv,
				scoring={
					"accuracy": "accuracy",
					"f1_macro": "f1_macro",
				},
				n_jobs=-1,
				error_score="raise",
				return_train_score=False,
			)

			# Fit on full train and evaluate on holdout test
			model.fit(X_train, y_train)
			y_test_pred = model.predict(X_test)
			test_acc = accuracy_score(y_test, y_test_pred)
			test_f1 = f1_score(y_test, y_test_pred, average="macro")

			model_report[name] = test_acc

			logging.info(
				f"{name}: CV acc={cv_scores['test_accuracy'].mean():.4f}, "
				f"CV f1_macro={cv_scores['test_f1_macro'].mean():.4f}, "
				f"Test acc={test_acc:.4f}, Test f1_macro={test_f1:.4f}"
			)

		return model_report

	except Exception as e:
		logging.error(f"Error occurred during model evaluation: {e}")
		raise CustomException(e, sys)

def load_hyperparameters(config_path: str = "params.yaml") -> dict:
	"""Loads hyperparameters from a YAML configuration file.
	
	Args:
		config_path (str): Path to the YAML file containing hyperparameters.
		
	Returns:
		dict: Dictionary containing hyperparameters for all models.
		
	Raises:
		CustomException: If there is an error loading the YAML file.
	"""
	try:
		with open(config_path, 'r') as file:
			params = yaml.safe_load(file)
		logging.info(f"Hyperparameters loaded from {config_path}")
		return params
	except Exception as e:
		logging.error(f"Error occurred while loading hyperparameters from {config_path}: {e}")
		raise CustomException(e, sys)

def get_model_params(params: dict, model_name: str) -> dict:
	"""Extracts and formats hyperparameters for a specific model.
	
	Separates grid search parameters from static parameters.
	
	Args:
		params (dict): Dictionary containing all hyperparameters.
		model_name (str): Name of the model.
		
	Returns:
		dict: Dictionary with grid parameters for GridSearchCV.
	"""
	model_key = model_name.lower().replace(" ", "_")
	if model_key not in params:
		return {}
	
	model_params = params[model_key]
	
	# Parameters that should not be in grid (will be set directly on model)
	non_grid_params = {"random_state", "tree_method", "objective", "eval_metric", 
	                    "num_class", "verbose"}
	
	# Extract only grid parameters (those with list values)
	grid_params = {}
	for key, value in model_params.items():
		if key not in non_grid_params and isinstance(value, list):
			grid_params[key] = value
	
	return grid_params