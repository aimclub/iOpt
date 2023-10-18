import data

from argparse import ArgumentParser
from dataclasses import dataclass, field

from hyperparams import Hyperparameter, Numerical, Categorial
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from functools import partial


METHOD_TO_HYPERPARAMS = {
    SVC: {
        'gamma': Numerical('float', 1e-9, 1e-6, is_log_scale=True),
        'C': Numerical('int', 1, 1e10, is_log_scale=True),
        'kernel': Categorial('poly', 'rbf', 'sigmoid')
    },

    XGBClassifier: {
        'n_estimators': Numerical('int', 10, 200),
        'max_depth': Numerical('int', 5, 20),
        'min_child_weight': Numerical('int', 1, 10),
        'gamma': Numerical('float', 0.01, 0.6),
        'subsample': Numerical('float', 0.05, 0.95),
        'colsample_bytree': Numerical('float', 0.05, 0.95),
        'learning_rate': Numerical('float', 0.001, 0.1, is_log_scale=True)
    },

    MLPClassifier: {
        'hidden_layer_sizes': Numerical('int', 2, 150),
        'activation': Categorial('identity', 'logistic', 'tanh', 'relu'),
        'solver': Categorial('lbfgs', 'sgd', 'adam'),
        'alpha': Numerical('float', 1e-9, 1e-1, is_log_scale=True)
    }
}


NAME_TO_DATASET = {
    'balance': data.Balance,
    'bank-marketing': data.BankMarketing,
    'banknote': data.Banknote,
    'breast-cancer': data.BreastCancer,
    'car-evaluation': data.CarEvaluation,
    'cnae9': data.CNAE9,
    'credit-approval': data.CreditApproval,
    'digits': data.Digits,
    'ecoli': data.Ecoli,
    'parkinsons': data.Parkinsons,
    'semeion': data.Semeion,
    'statlog-segmentation': data.StatlogSegmentation,
    'wilt': data.Wilt,
    'zoo': data.Zoo
}


@dataclass
class ConsoleArgument:
    max_iter: int
    estimator: SVC | XGBClassifier | MLPClassifier
    dataset: data.Dataset
    hyperparams: Hyperparameter = field(init=False)
    dir: str
    trials: int
    n_jobs: int

    def __post_init__(self):
        estimator = self.estimator
        if isinstance(estimator, partial):
            estimator = estimator.func
        self.hyperparams = METHOD_TO_HYPERPARAMS[estimator]


def get_estimator(name: str) -> SVC | XGBClassifier | MLPClassifier:
    if name == 'svc':
        return partial(SVC, max_iter=1000)
    elif name == 'xgb':
        return partial(XGBClassifier, n_jobs=1)
    elif name == 'mlp':
        return MLPClassifier
    raise ValueError(f'Estimator "{name}" do not support')


def get_datasets(names: str) -> data.Dataset:
    try:
        result = []
        for x in names:
            result.append(NAME_TO_DATASET[x])
        return result
    except KeyError:
        raise ValueError(f' Dataset "{x}" do not support')


def parse_arguments():
    """
    --max-iter:
        int, positive
    --dataset:
        names of dataset, see all names in NAME_TO_DATASET dict
    --method:
        must be or svc, or xgb, or mlp
    --dir:
        name of the dir to save the results (result by default)

    """
    parser = ArgumentParser()
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--dataset', nargs='*')
    parser.add_argument('--method')
    parser.add_argument('--dir', default='result')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--n-jobs', type=int, default=1)
    
    args = parser.parse_args()
    assert args.max_iter > 0, 'Max iter must be positive'
    assert args.trials > 0, 'Trials must be positive'
    assert args.n_jobs > 0, 'n_jobs must be positive'

    return ConsoleArgument(args.max_iter,
                           get_estimator(args.method),
                           get_datasets(args.dataset),
                           args.dir, args.trials, args.n_jobs)
