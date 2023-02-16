from typing import List

from EasyChemML.JobSystem.Utilities.Config import Config


class ConfigStack:
    _configs: List[Config]

    def __init__(self, configs: List[Config]):
        self._configs = configs

    def __iter__(self):
        return self._configs.__iter__()
