import os
import numpy as np

from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.MetricsList import MetricList

from prettytable import from_csv

from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.R2_Score import R2_Score


def test_saveMetricListAsCSV():
    ml: MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()
    m3 = initMetrics()
    m4 = initMetrics()

    ml + m1
    ml + m2
    ml + m3
    ml + m4

    ml.saveMetricAsCSV('test.csv')

    with open('test.csv', 'r', newline='', encoding='utf-8') as fp:
        x = from_csv(fp)

    print(x)
    os.remove('test.csv')


def test_calcAverage():
    ml: MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()
    m3 = initMetrics()
    m4 = initMetrics()

    ml + m1
    ml + m2
    ml + m3
    ml + m4

    ava_metric = ml.calcAverage()

    for key in ava_metric:
        ava = np.average([m1[key], m2[key], m3[key], m4[key]])
        if not ava_metric[key] == ava:
            assert False

    assert True

def test_getBestMetric():
    ml: MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()
    m3 = initMetrics()
    m4 = initMetrics()

    ml + m1
    ml + m2
    ml + m3
    ml + m4

    m2.metric_data['r2_score'] = 1

    if not m2 == ml.getbestMetric('r2_score'):
        assert False

    m1.metric_data['MAE'] = 0

    if not m1 == ml.getbestMetric('MAE'):
        assert False

    assert True


def test_getitem():
    ml:MetricList = MetricList()
    m1 = initMetrics()
    m2 = initMetrics()

    ml + m1
    ml + m2

    assert ml[0] == m1
    assert ml[1] == m2
    assert not ml[1] == m1

def test_add():
    ml:MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()

    ml + m1

    if not ml[0] == m1:
        assert False

    ml + m2

    if not ml[1] == m2:
        assert False

    assert True

def test_delitem():
    ml:MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()
    m3 = initMetrics()
    m4 = initMetrics()

    ml + m1
    ml + m2
    ml + m3
    ml + m4

    if not (ml[0] == m1 and ml[1] == m2 and ml[2] == m3 and ml[3] == m4):
        assert False

    del ml[0]

    if not (ml[0] == m2 and ml[1] == m3 and ml[2] == m4):
        assert False

    del ml[1]

    if not (ml[0] == m2 and ml[1] == m4):
        assert False

    del ml[1]

    if not (ml[0] == m2):
        assert False

    del ml[0]

    if not len(ml) == 0:
        assert False

    assert True

def test_len():
    ml:MetricList = MetricList()

    m1 = initMetrics()
    m2 = initMetrics()
    m3 = initMetrics()
    m4 = initMetrics()

    if not len(ml) == 0:
        assert False

    ml + m1

    if not len(ml) == 1:
        assert False

    ml + m2

    if not len(ml) == 2:
        assert False

    ml + m3

    if not len(ml) == 3:
        assert False

    ml + m4

    if not len(ml) == 4:
        assert False

    assert True

def initMetrics():
    trues = np.random.randn(100)
    predicted = np.random.randn(100)

    r2score = R2_Score()
    mae = MeanAbsoluteError()

    m = MetricStack({'r2_score': r2score, 'MAE': mae})
    m.calcMetric(trues, predicted)

    return m