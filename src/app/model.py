import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import time

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Model:
    """Model class for training and predicting - Linear Regression as an example"""

    @dataclass()
    class Params:
        """Class for model parameters"""
        slope: float
        intercept: float

    rs = 42  # random state

    def __init__(self) -> None:
        self.model = LinearRegression()

    def load_data(self, path: Path) -> None:
        """loads data from the path - actually just generates random data for the example"""
        self.path = path
        self.data = pd.read_csv(self.path)
        self.X = pd.DataFrame(np.random.randint(0, 40, size=(10, 1)), columns=['x'])
        self.y = pd.DataFrame(np.random.randint(0, 40, size=(10, 1)), columns=['y'])
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, random_state=Model.rs)

    def get_model_parameters(self) -> LinearRegression:
        """returns the trained model"""
        return Model.Params(self.model.coef_, self.model.intercept_)

    def train(self) -> None:
        """starts training the model"""
        time.sleep(3)  # simulate long training time and show that the GUI is not frozen
        self.model.fit(self.train_X, self.train_y)

    def predict(self, x: float) -> float:
        """predicts the model"""
        return self.model.predict(x)

    def predict_multi(self, x: pd.DataFrame) -> pd.DataFrame:
        """predicts the model"""
        return self.model.predict(x)

    def validate(self) -> float:
        """validates the model and returns the mean absolute error"""
        predictions = self.model.predict(self.val_X)
        return mean_absolute_error(predictions, self.val_y)
