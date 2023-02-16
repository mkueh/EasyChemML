from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from EasyChemML.Utilities.Application_env import Application_env


class NoneAbstractEncoder(AbstractEncoder):

    def __init__(self, APP_ENV:Application_env):
        super().__init__(APP_ENV)

    """
    Parameter
    coulmns: defines the coulumns which are converted to RDKITMol
    nanvalue: defines the NAN value for that is used in the dataset
    """
    def convert(self, dataset, n_jobs:int, **kwargs):
        return dataset.getFeature_data()

    @staticmethod
    def getItemname():
        return "e_NONE"