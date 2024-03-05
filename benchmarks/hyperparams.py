from abc import ABC
from dataclasses import dataclass, field
from typing import Literal


class Hyperparameter(ABC):
    pass


@dataclass
class Numerical(Hyperparameter):
    type: Literal['float', 'int']
    min_value: float
    max_value: float
    is_log_scale: bool = False


@dataclass
class Categorial(Hyperparameter):
    values: tuple[str] = field(init=False)

    def __init__(self, *args):
        self.values = args
