import numpy as np

from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError


def test_metricStack_easy():
    r2score = R2_Score()
    mae = MeanAbsoluteError()

    metricStack = MetricStack({'r2_score': r2score, 'MAE': mae})

    trues = np.random.randn(100)
    predicted = np.random.randn(100)

    metricStack.calcMetric(trues, predicted)
    print(metricStack)


def test_metricStack_equal():
    r2score = R2_Score({'test':'test'})
    metricStack_1 = MetricStack({'r2_score': r2score})

    r2score_2 = R2_Score({'test':'test'})
    metricStack_2 = MetricStack({'r2_score': r2score_2})

    metricStack_3 = MetricStack({'r2_score': r2score})

    r2score_4 = R2_Score({'test':'test2'})
    metricStack_4 = MetricStack({'r2_score': r2score_4})

    r2score_5 = R2_Score({'test':'test', 'test_1':12})
    metricStack_5 = MetricStack({'r2_score': r2score_5})

    assert metricStack_1.equal_config(metricStack_2)
    assert metricStack_2.equal_config(metricStack_3)
    assert metricStack_1.equal_config(metricStack_3)
    assert not metricStack_1.equal_config(metricStack_4)
    assert not metricStack_3.equal_config(metricStack_4)
    assert not metricStack_4.equal_config(metricStack_5)
    assert not metricStack_1.equal_config(metricStack_5)