import os
from datetime import datetime
from enum import Enum

import shutil
from typing import Dict, Any, Tuple


class Checkpoint:
    timestamp: float
    meta_path: str
    data_path: str

    def __init__(self, meta_path: str, data_path: str):
        self.timestamp = datetime.now().timestamp()
        self.meta_path = meta_path
        self.data_path = data_path

    def __eq__(self, other):
        if isinstance(other, Checkpoint):
            if self.meta_path == other.meta_path and self.data_path == other.data_path:
                return True
            return False
        else:
            super.__eq__(self, other)

    def __str__(self):
        return self.getCheckpointName()

    def getCheckpointName(self) -> str:
        head, tail = os.path.split(self.data_path)
        return tail

    def remove(self):
        os.remove(self.meta_path)
        os.remove(self.data_path)

    def getstate(self):
        state = {}

        state['timestamp'] = self.timestamp
        state['meta_path'] = self.meta_path
        state['data_path'] = self.data_path

        return state

    def setstate(self, state:Dict[str, Any]):
        self.timestamp = state['timestamp']
        self.meta_path = state['meta_path']
        self.data_path = state['data_path']


class ModelCheckpoint:
    pass
