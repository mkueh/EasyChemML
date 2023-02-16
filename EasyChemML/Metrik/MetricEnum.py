from enum import Enum, auto

class MetricClass(Enum):
    classification = 'clas'
    regression = 'regr'
    clustering = 'clust'
    generativ = 'gen'

class MetricType(Enum):
    absolute = 'abs'
    relative = 'rel'
    unkown = 'unk'

class MetricDirection(Enum):
    higherIsBetter = 'h'
    lowerIsBetter = 'l'
    oneIsBest = 'o'
    mixed = 'm'

class MetricOutputType(Enum):
    singleValue = 'sV'
    multiValue = 'mV'
    matrixValue = 'matV'