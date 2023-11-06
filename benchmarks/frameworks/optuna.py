import optuna
from functools import partial

from hyperparams import Hyperparameter, Numerical, Categorial
from .interface import Searcher, Point


ALGORITHMS = {
    'random': optuna.samplers.RandomSampler,
    'tpe': optuna.samplers.TPESampler,
    'cmaes': optuna.samplers.CmaEsSampler,
    'nsgaii': optuna.samplers.NSGAIISampler,
}


class OptunaSearcher(Searcher):
    def __init__(self, max_iter, *, algorithm: str = 'tpe', is_deterministic=False):
        super().__init__(framework_name='Optuna',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)

        self.algorithm, self.func_algorithm = algorithm, ALGORITHMS[algorithm]
        self.algorithm_kwargs = {}
        if self.algorithm == 'cmaes':
            self.algorithm_kwargs['warn_independent_sampling'] = False

    def _get_points(self):
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='maximize',
                                    sampler=self.func_algorithm(**self.algorithm_kwargs))

        points = []
        objective = partial(self.__objective, points=points)
        study.optimize(objective, n_trials=self.max_iter)
        return points

    def _get_searcher_params(self):
        return {'algorithm': self.algorithm}

    def __objective(self, trial: optuna.Trial, points: list[Point]):
        arguments = {}
        for name, p in self.hyperparams.items():
            arguments[name] = self.__get_suggest(name, p, trial)

        point = self._calculate_metric(arguments)
        points.append(point)
        return point.value

    @staticmethod
    def __get_suggest(name: str, p: Hyperparameter, trial: optuna.Trial):
        functions = {
            'int': trial.suggest_int,
            'float': trial.suggest_float
        }
        if isinstance(p, Numerical):
            return functions[p.type](name, p.min_value, p.max_value, log=p.is_log_scale)
        elif isinstance(p, Categorial):
            return trial.suggest_categorical(name, p.values)
