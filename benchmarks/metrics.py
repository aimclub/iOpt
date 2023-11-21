from abc import ABC, abstractclassmethod
from typing import Literal

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import data


PREPROCESSING = {
    'standard': StandardScaler,
    'min_max': MinMaxScaler,
    'max_abs': MaxAbsScaler,
}


class Metric(ABC):
    def __init__(self, name, preprocessing=None, is_regression=False):
        self.name = name
        self.preprocessing = PREPROCESSING[preprocessing] if (preprocessing is not None) else None
        if is_regression:
            self.cv = KFold(shuffle=True, random_state=42)
        else:
            self.cv = StratifiedKFold(shuffle=True, random_state=42)

    def __call__(self, estimator, dataset: data.Dataset):

        if self.preprocessing is not None:
            pipeline = Pipeline([
                ('scaler', self.preprocessing()),
                ('model', estimator)
            ])
        else:
            pipeline = estimator

        return cross_val_score(pipeline, dataset.features, dataset.targets,
                               scoring=self.get_score, cv=self.cv).mean()

    def log_params(self):
        return {'name': self.name,
                'preprocessing': self.preprocessing.__name__ if self.preprocessing else None}

    @abstractclassmethod
    def get_score(self, model, x, y):
        pass


class Accuracy(Metric):
    def __init__(self, preprocessing):
        super().__init__('accuracy', preprocessing)

    def get_score(self, model, x, y):
        return accuracy_score(y, model.predict(x))


class F1(Metric):
    def __init__(self, preprocessing, average: Literal['binary', 'macro', 'micro']):
        super().__init__(f'f1-score -> {average}', preprocessing)
        self.average = average

    def get_score(self, model, x, y):
        return f1_score(y, model.predict(x), average=self.average)


class R2(Metric):
    def __init__(self, preprocessing):
        super().__init__('r2-score', preprocessing, is_regression=True)
    
    def get_score(self, model, x, y):
        return r2_score(y, model.predict(x))


DATASET_TO_METRIC = {
    data.BreastCancer:               F1(preprocessing='standard', average='binary'),
    data.Digits:                     Accuracy(preprocessing='standard'),
    data.BankMarketing:              Accuracy(preprocessing='standard'),
    data.CNAE9:                      Accuracy(preprocessing='standard'),
    data.StatlogSegmentation:        Accuracy(preprocessing='standard'),
    data.Semeion:                    Accuracy(preprocessing='standard'),
    data.Ecoli:                      F1(preprocessing='standard', average='macro'),
    data.CreditApproval:             F1(preprocessing='standard', average='binary'),
    data.Balance:                    F1(preprocessing='standard', average='macro'),
    data.Parkinsons:                 F1(preprocessing='standard', average='binary'),
    data.Zoo:                        F1(preprocessing='standard', average='macro'),
    data.Banknote:                   F1(preprocessing='standard', average='binary'),
    data.CarEvaluation:              F1(preprocessing='standard', average='macro'),
    data.Wilt:                       F1(preprocessing='standard', average='binary'),
    data.Transformator:              F1(preprocessing=None, average='macro'),
    data.Turbine:                    R2(preprocessing=None)
}
