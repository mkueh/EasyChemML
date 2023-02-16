import csv, copy
from typing import List
import numpy as np

from .MetricStack import MetricStack
from .MetricEnum import MetricDirection


class MetricList(object):
    metrics: List[MetricStack] = -1

    def __init__(self):
        self.metrics = []

    def __getitem__(self, index):
        return self.metrics[index]

    def __add__(self, other: MetricStack):
        if isinstance(other, MetricStack):
            if len(self.metrics) > 0:
                if self.metrics[0].equal_config(other):
                    self.metrics.append(other)
                else:
                    raise Exception('It is not possible to add a Metric with a different metricset to the metric list')
            else:
                self.metrics.append(other)
        else:
            raise Exception('Cant add a none Metric-Type to MetricList')
        return self

    def __delitem__(self, index):
        del self.metrics[index]

    def __len__(self):
        return len(self.metrics)

    def calcAverage(self) -> MetricStack:
        avarage_dict = {}

        if len(self.metrics) > 0:
            possiblemetrics = self.metrics[0].keys()

            for metric_name in possiblemetrics:
                listofValues = []
                for metric in self.metrics:
                    listofValues.append(metric[metric_name])

                avarage_dict[metric_name] = np.average(listofValues)

            ava_metric = MetricStack(copy.deepcopy(self.metrics[0].metric_modules))
            ava_metric.metric_data = avarage_dict
            return ava_metric
        else:
            raise Exception('The MetricList ist empty')

    def getbestMetric(self, metric_name: str, returnIndex:bool = False):
        if len(self.metrics) > 0:
            value = 0
            best_index = 0
            metric_module = self.metrics[0].getModule(metric_name)

            for index, m in enumerate(self.metrics):
                # init values with first
                if index == 0:
                    value = self.metrics[index][metric_name]
                    best_index = index
                else:
                    direction = metric_module.getDirection()
                    if direction == MetricDirection.higherIsBetter:
                        if value < self.metrics[index][metric_name]:
                            value = self.metrics[index][metric_name]
                            best_index = index
                    elif direction == MetricDirection.lowerIsBetter:
                        if value > self.metrics[index][metric_name]:
                            value = self.metrics[index][metric_name]
                            best_index = index
                    elif direction == MetricDirection.oneIsBest:
                        if abs(1 - value) > abs(1 - self.metrics[index][metric_name]):
                            value = self.metrics[index][metric_name]
                            best_index = index
                    else:
                        raise Exception('getbestMetric is not possible when direction is mixed')
            if not returnIndex:
                return self.metrics[best_index]
            else:
                return best_index
        else:
            raise Exception('The MetricList ist empty')

    def saveMetricAsCSV(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            ava_metric = self.calcAverage()

            writer.writerow(['Average'])
            writer.writerow([k for k in ava_metric.metric_modules])
            writer.writerow([v for v in ava_metric.values()])

            writer.writerow(['Steps'])

            writer.writerow(['NR.'] + [k for k in ava_metric.metric_modules])

            for i, m in enumerate(self.metrics):
                rowList = [str(i)]
                for value in m.values():
                    rowList.append(value)
                writer.writerow(rowList)
