import hyperopt
import numpy as np

from functools import partial
from hyperparams import Numerical, Categorial
from .interface import Searcher, Point


ALGORITHMS = {
    'random': hyperopt.rand.suggest,
    'tpe': hyperopt.tpe.suggest,
    'anneal': hyperopt.anneal.suggest,
}


class HyperoptSearcher(Searcher):
    def __init__(self, max_iter, *, algorithm: str = 'tpe', is_deterministic=False):
        super().__init__(framework_name='Hyperopt',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)

        self.algorithm, self.func_algorithm = algorithm, ALGORITHMS[algorithm]
        self.numerical_func = {
            'float': [hyperopt.hp.uniform, hyperopt.hp.loguniform],
            'int': [partial(hyperopt.hp.quniform, q=1), partial(hyperopt.hp.qloguniform, q=1)]
        }

    def _get_searcher_params(self):
        return {'algorithm': self.algorithm}

    def __objective(self, arguments, points: list):
        self.__float_to_int(arguments)
        point = self._calculate_metric(arguments)
        points.append(point)
        return -point.value

    def __get_hyperparam_space(self):
        space = {}
        for name, p in self.hyperparams.items():
            if isinstance(p, Numerical):
                space[name] = self.numerical_func[p.type][p.is_log_scale](name,
                    np.log(p.min_value) if p.is_log_scale else p.min_value,
                    np.log(p.max_value) if p.is_log_scale else p.max_value)
            elif isinstance(p, Categorial):
                space[name] = hyperopt.hp.choice(name, p.values)
        return space

    def __float_to_int(self, arguments: dict):
        for name, value in arguments.items():
            x = self.hyperparams[name]
            if isinstance(x, Numerical) and x.type == 'int':
                arguments[name] = int(value + 0.5)

    def _get_points(self) -> list[Point]:
        space = self.__get_hyperparam_space()
        points = []
        objective = partial(self.__objective, points=points)
        hyperopt.fmin(objective, space,
                      max_evals=self.max_iter, trials=hyperopt.Trials(),
                      verbose=False, algo=self.func_algorithm)
        return points
