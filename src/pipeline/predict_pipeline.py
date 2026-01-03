import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """Pipeline for making predictions using the trained model."""
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
    def predict(self, features):
        """
        Make predictions using the trained model.
        
        Args:
            features (pd.DataFrame): Input features for prediction.
            
        Returns:
            np.ndarray: Predictions.
        """
        try:
            logging.info("Loading model and preprocessor for prediction")
            
            # Load the trained model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            logging.info("Preprocessing input features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions")
            predictions = model.predict(data_scaled)
            
            logging.info("Predictions completed successfully")
            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    """Class for handling custom input data."""
    
    def __init__(
        self,
        Age: float,
        Height: float,
        Weight: float,
        FCVC: float,
        NCP: float,
        CH2O: float,
        FAF: float,
        TUE: float,
        Gender: str,
        family_history_with_overweight: str,
        FAVC: str,
        CAEC: str,
        SMOKE: str,
        SCC: str,
        CALC: str,
        MTRANS: str
    ):
        """
        Initialize custom data with input features.
        
        Args:
            Age (float): Age of the person
            Height (float): Height in meters
            Weight (float): Weight in kilograms
            FCVC (float): Frequency of consumption of vegetables
            NCP (float): Number of main meals
            CH2O (float): Daily water consumption
            FAF (float): Physical activity frequency
            TUE (float): Time using technology devices
            Gender (str): Gender (Male/Female)
            family_history_with_overweight (str): Family history (yes/no)
            FAVC (str): Frequent consumption of high caloric food (yes/no)
            CAEC (str): Consumption of food between meals
            SMOKE (str): Smoking status (yes/no)
            SCC (str): Calories consumption monitoring (yes/no)
            CALC (str): Consumption of alcohol
            MTRANS (str): Transportation used
        """
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.FCVC = FCVC
        self.NCP = NCP
        self.CH2O = CH2O
        self.FAF = FAF
        self.TUE = TUE
        self.Gender = Gender
        self.family_history_with_overweight = family_history_with_overweight
        self.FAVC = FAVC
        self.CAEC = CAEC
        self.SMOKE = SMOKE
        self.SCC = SCC
        self.CALC = CALC
        self.MTRANS = MTRANS
        
    def get_data_as_dataframe(self):
        """
        Convert the custom data to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Input data as a DataFrame.
        """
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "FCVC": [self.FCVC],
                "NCP": [self.NCP],
                "CH2O": [self.CH2O],
                "FAF": [self.FAF],
                "TUE": [self.TUE],
                "Gender": [self.Gender],
                "family_history_with_overweight": [self.family_history_with_overweight],
                "FAVC": [self.FAVC],
                "CAEC": [self.CAEC],
                "SMOKE": [self.SMOKE],
                "SCC": [self.SCC],
                "CALC": [self.CALC],
                "MTRANS": [self.MTRANS]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data converted to DataFrame")
            return df
            
        except Exception as e:
            logging.error(f"Error converting custom data to DataFrame: {str(e)}")
            raise CustomException(e, sys)
