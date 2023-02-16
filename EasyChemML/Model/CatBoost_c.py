from catboost import CatBoostClassifier

from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model
from EasyChemML.Metrik.MetricStack import MetricClass
import numpy as np


class CatBoost_c(Abstract_Model):
    clf = None

    # https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

    def __init__(self, param=None, log_folder: str = ''):
        super().__init__()
        if param is None:
            param = {}
        self.clf = CatBoostClassifier(**param)

    def set_param(self, param: dict):
        self.clf.set_params(**param)

    def get_param(self) -> dict:
        return self.clf.get_params()

    def fit(self, X: np.ndarray, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predicte_proba(self, X: np.ndarray, **kwargs):
        """
        Get predicted probabilities for each class
        """
        return self.clf.predict_proba(X)

    def save_model(self, path: str):
        self.clf.save_model(path)

    def load_model(self, path: str):
        self.clf.load_model(path)

    @staticmethod
    def getMetricMode():
        return MetricClass.classification

    @staticmethod
    def hasBatchMode():
        return False

    @staticmethod
    def hasPredicte_proba():
        return True
