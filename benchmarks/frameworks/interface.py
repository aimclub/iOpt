from abc import ABC, abstractclassmethod

from metrics import Metric
from data.loader import Dataset
from hyperparams import Hyperparameter, Categorial
from dataclasses import dataclass
from time import time


@dataclass
class Point:
    timepoint: int
    value: float
    params: dict[str]


class Searcher(ABC):
    def __init__(self,
                 framework_name: str, *, max_iter: int, is_deterministic: bool):

        self.framework_name = framework_name
        self.max_iter = max_iter
        self.is_deterministic = is_deterministic

    @abstractclassmethod
    def _get_points() -> list[Point]:
        pass

    @abstractclassmethod
    def _get_searcher_params(self) -> dict:
        pass

    def __str__(self) -> str:
        params = self._get_searcher_params()
        params['max_iter'] = self.max_iter
        params['is_deterministic'] = self.is_deterministic
        arguments = ', '.join(f'{key}={value}' for key, value in params.items())
        return f'{self.framework_name}({arguments})'

    def tune(self, estimator,
                   hyperparams: dict[str, Hyperparameter],
                   dataset: Dataset,
                   metric: Metric) -> list[Point]:
        self.estimator, self.hyperparams, self.dataset, self.metric = estimator, hyperparams, dataset, metric
        return self._get_points()

    def _calculate_metric(self, arguments: dict):
        model = self.estimator(**arguments)
        value = self.metric(model, self.dataset)
        timepoint = int(time() * 1000)

        for name_param, value_param in arguments.items():
            x = self.hyperparams[name_param]
            if isinstance(x, Categorial):
                arguments[name_param] = x.values.index(value_param)

        return Point(timepoint, value, arguments)
