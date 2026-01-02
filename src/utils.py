import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

import dill
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

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