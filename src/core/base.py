from typing import Dict
from abc import ABCMeta, abstractmethod


class BaseDict(Dict):
    """simple implementation of attribute dict"""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.__dict__ = self

    def copy(self, *args, **kwargs):
        _obj = super(BaseDict, self).copy()
        new_obj = BaseDict()
        new_obj.update(_obj)
        return new_obj


class BaseAttacker(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, *args, **kwargs):
        pass


class Condition(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass
