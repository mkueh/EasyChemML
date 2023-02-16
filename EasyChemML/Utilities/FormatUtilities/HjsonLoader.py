import os, hjson
from typing import Dict


class HjsonLoader:

    @staticmethod
    def load_Hjson(path: str) -> dict:
        with open(path, 'r', encoding='utf8') as json_file:
            data = hjson.load(json_file)
        return data

    @staticmethod
    def dump_Hjson(data: Dict, path: str):
        with open(path, 'w', encoding='utf8') as json_file:
            hjson.dump(data, json_file)
