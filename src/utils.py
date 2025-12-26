import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

import dill

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