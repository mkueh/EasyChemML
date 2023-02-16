from abc import ABC, abstractmethod

from EasyChemML.Environment import Environment


class Abstract_Hyperparametersearch():
    __env: Environment

    def __init__(self, env:Environment):
        self.__env = env
