from sklearn.ensemble import RandomForestClassifier
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model
import numpy as np

from EasyChemML.Utilities.CompressUtilities.lz4_compressor import lz4_compressor


class scikit_RandomForestClassifier(Abstract_Model):
    clf = None

    def __init__(self, param = None, log_folder: str  = ''):
        super().__init__()
        if param is None:
            param = {}
        self.clf = RandomForestClassifier(**param)

    def load_model(self, path: str):
        lz4 = lz4_compressor()
        self.clf = lz4.decompress_object_from_file(path)

    def save_model(self, path: str):
        lz4 = lz4_compressor()
        lz4.compress_object_to_file(self.clf, path)

    def set_param(self, param: dict):
        self.clf.set_params(**param)

    def get_param(self) -> dict:
        return self.clf.get_params()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predicte_proba(self, X: np.ndarray, **kwargs):
        """
        Get predicted probabilities for each class
        """
        return self.clf.predict_proba(X)

    @staticmethod
    def getMetricMode():
        return MetricClass.classifier

    def hasPredicte_proba(self):
        return True

    @staticmethod
    def hasBatchMode():
        return False
