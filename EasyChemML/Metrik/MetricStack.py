import copy
from typing import Dict, Union, Optional

import deepdiff

from EasyChemML.Metrik.Module.Abstract_Metric import Abstract_Metric
from .MetricEnum import MetricClass


class MetricStack(object):
    metric_modules: Dict[str, Abstract_Metric]
    metric_data: Dict

    def __init__(self, metric_modules_named: Dict[str, Abstract_Metric] = {}, metric_data: Dict = {}):
        self.metric_modules = metric_modules_named
        self.metric_data = metric_data

        if len(self.metric_modules) <= 0:
            raise Exception('No metric is set')

        tmp_holder: MetricClass = -1
        for metric in self.metric_modules:
            if tmp_holder == -1:
                tmp_holder = self.metric_modules[metric].getMetricClass()
            elif not self.metric_modules[metric].getMetricClass() == tmp_holder:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('MetricStack is in mixed mode because a mix of regressor, classifier and clustering is selected')
                print('this is not recommended because some metrics can not process continuous values')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break
            else:
                pass

    def __repr__(self):
        string = ''

        if len(self.metric_data) > 0:
            for i, name in enumerate(self.metric_modules):
                if i == 0:
                    string = f'{name}: {self.metric_data[name]}'
                else:
                    string = string + f' | {name}: {self.metric_data[name]}'
            return string
        else:
            for i, name in enumerate(self.metric_modules):
                if i == 0:
                    string = f'{name}: not calculated yet'
                else:
                    string = string + f' | {name}: not calculated yet'
            return string

    def __getitem__(self, item):
        return self.metric_data.__getitem__(item)

    def getModule(self, item) -> Abstract_Metric:
        return self.metric_modules.__getitem__(item)

    def __eq__(self, other):
        if isinstance(other, MetricStack):
            result = len(deepdiff.DeepDiff(self.metric_data, other.metric_data)) == 0
            return result
        else:
            return False

    def equal_config(self, other):
        if isinstance(other, MetricStack):
            for my_module_name in self.metric_modules:
                if my_module_name in other.metric_modules:
                    if not self.getModule(my_module_name) == other.getModule(my_module_name):
                        return False
                else:
                    return False
            return True
        else:
            return False

    def __iter__(self):
        return self.metric_data.__iter__()

    def values(self):
        values = []

        for entry_key in self.metric_data:
            values.append(self.metric_data[entry_key])

        return values

    def keys(self):
        return self.metric_modules.keys()

    def keys_asList(self):
        return list(self.keys())

    def __copy__(self):
        def __deepcopy__(self, memodict={}):
            """
                copy of metric moduls and deepcopy metric data
            """
        return MetricStack(self.metric_modules, copy.deepcopy(self.metric_data))

    def __deepcopy__(self, memodict={}):
        """
            deepcopy of metric moduls and metric data
        """
        return MetricStack(copy.deepcopy(self.metric_modules), copy.deepcopy(self.metric_data))

    def calcMetric(self, y_true, y_predict, y_predict_proba=None) -> 'MetricStack':
        metric_data = {}
        for i, name in enumerate(self.keys_asList()):
            metric_module = self.metric_modules[name]
            metric_data[name] = metric_module.calc(y_true, y_predict, y_predict_proba)
        return MetricStack(self.metric_modules, metric_data)
