from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from functools import partial

import optuna
import hyperopt
import pandas as pd

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


def get_score(model, x, y):
    return cross_val_score(model, x, y, cv=5, n_jobs=-1,
                           scoring=lambda model, x, y: f1_score(model.predict(x), y, average='weighted')).mean()


class Searcher(ABC):
    def __init__(self, method, params, datasets, max_iter):
        self.method = method
        self.params = params
        self.datasets = datasets
        self.max_iter = max_iter
    
    @abstractmethod
    def hyperparams_tune(self):
        pass


class OptunaSearch(Searcher):
    def __init__(self, method, params, datasets, max_iter):
        super().__init__(method, params, datasets, max_iter)


    def hyperparams_tune(self):
        optuna.logging.disable_default_handler()
        result = {}
        for name, x, y in self.datasets:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, x, y), 
                           n_trials=self.max_iter)
            result[name] = study.best_value
        return result

    
    def objective(self, trial, x, y):
        params = {}
        for name, (type, min_, max_) in self.params.items():
            if type is float:
                params[name] = trial.suggest_float(name, min_, max_)
            elif type is int:
                params[name] = trial.suggest_int(name, min_, max_)
        model = self.method(**params)
        return get_score(model, x, y)


class HyperOptSearch(Searcher):
    def __init__(self, method, params, datasets, max_iter):
        super().__init__(method, params, datasets, max_iter)
    
    def hyperparams_tune(self):
        params = {}
        for name, (type, min_, max_) in self.params.items():
            if type is float:
                params[name] = (type, hyperopt.hp.uniform(name, min_, max_))
            elif type is int:
                params[name] = (type, hyperopt.hp.quniform(name, min_, max_, q=1))

        result = {}
        for name, x, y in self.datasets:
            trial = hyperopt.Trials()
            fn = partial(self.objective, x=x, y=y)
            hyperopt.fmin(fn=fn,
                          space=params,
                          algo=hyperopt.tpe.suggest,
                          max_evals=self.max_iter, 
                          trials=trial, 
                          verbose=False)
            result[name] = -trial.best_trial['result']['loss']
        return result

    def objective(self, params, x, y):
        model = self.method(**{name: int(value) if type is int else value for name, (type, value) in params.items()})
        return -get_score(model, x, y)


class SkicitLearnMethod(Problem):
    def __init__(self, x, y, method, params):
        super().__init__()

        self.variable_type = [t for t, _, _ in params.values()]

        self.numberOfFloatVariables = len(params)
        self.dimension = len(params)
        self.numberOfObjectives = 1

        for name, (_, min_, max_) in params.items():

            self.floatVariableNames.append(name)
            self.lowerBoundOfFloatVariables.append(min_)
            self.upperBoundOfFloatVariables.append(max_)

        self.method = method
        self.x, self.y = x, y
    
    def Calculate(self, point, functionValue):
        arguments = {}
        for type, name, value in zip(self.variable_type, self.floatVariableNames, point.floatVariables):
            arguments[name] = int(value) if type is int else value
        method = self.method(**arguments)
        functionValue.value = -get_score(method, self.x, self.y)
        return functionValue


class iOptSearch(Searcher):
    def __init__(self, method, params, datasets, max_iter):
        super().__init__(method, params, datasets, max_iter)

    def hyperparams_tune(self):
        result = {}
        for name, x, y in self.datasets:
            problem = SkicitLearnMethod(x, y, self.method, self.params)
            method_params = SolverParameters(itersLimit=self.max_iter)
            solver = Solver(problem, parameters=method_params)
            solver_info = solver.Solve()
            result[name] = -solver_info.bestTrials[0].functionValues[-1].value
        return result


ALL_SEARCHERS = [OptunaSearch, HyperOptSearch, iOptSearch]


def compare_methods(method, params, datasets, max_iter):
    frame = pd.DataFrame(index=[x[0] for x in datasets])
    for searcher in ALL_SEARCHERS:
        instance = searcher(method, params, datasets, max_iter)
        frame[searcher.__name__] = instance.hyperparams_tune()
    return frame

