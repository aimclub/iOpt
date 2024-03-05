import requests
import zipfile
import os
import numpy as np
import numpy.typing as npt
import arff

from urllib.parse import urlencode
from abc import ABC, abstractclassmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer, load_digits
from pathlib import Path
from dataclasses import dataclass

from typing import List


@dataclass
class Dataset:
    name: str
    features: npt.NDArray
    targets: npt.NDArray


class Parser(ABC):
    def __init__(self, path):
        self.path = Path(__file__).parent / 'datasets' / path

    @abstractclassmethod
    def load_data(self, sample_skip=None) -> List[List]:
        pass

    @abstractclassmethod
    def load_dataset(self) -> Dataset:
        pass

    def preprocess_feature(self, feature):
        try:
            return [float(value) for value in feature]
        except ValueError:
            return LabelEncoder().fit_transform(feature).tolist()

    @staticmethod
    def feature_skip_condition(i, skip):
        return (isinstance(skip, int) and i != skip) or \
               (isinstance(skip, list) and i not in skip)

    def preprocess_features(self, features, skip=None):
        number_features = len(features[0])
        return np.array(
            [self.preprocess_feature([sample[i] for sample in features])
             for i in range(number_features) if (skip is None) or self.feature_skip_condition(i, skip)]
        ).T

    def preprocess_target(self, targets):
        return LabelEncoder().fit_transform(targets)

    @staticmethod
    def separate_target(data, index_target: int | List[int]):
        if isinstance(index_target, int):
            index_target = [index_target]
        for i, x in enumerate(index_target):
            index_target[i] = len(data[0]) + x if x < 0 else x
        features, targets = [], []
        for sample in data:
            features.append([x for i, x in enumerate(sample) if i not in index_target])
            target = [sample[i] for i in index_target]
            if len(target) == 1:
                targets.append(target[0])
            else:
                targets.append(target.index('1'))
        return features, targets

    def parse_file(self, index_target, sample_skip=None, feature_skip=None):
        data = self.load_data(sample_skip)
        features, targets = self.separate_target(data, index_target)
        return self.preprocess_features(features, skip=feature_skip), \
            self.preprocess_target(targets)


class TextParser(Parser):
    def __init__(self, path, spliter, skips='?'):
        super().__init__(f'uci-datasets/{path}')
        self.spliter = spliter
        self.skips = skips

    def load_data(self, sample_skip=None):
        with open(self.path) as f:
            data = []
            sample = -1
            while (row := f.readline()).strip():
                sample += 1
                if (sample_skip is not None) and (sample == sample_skip):
                    continue
                input_ = [x.strip() for x in row.split(self.spliter)]
                if self.skips in input_:
                    continue
                data.append(input_)
        return data


class ArffParser(Parser):
    def __init__(self, path):
        super().__init__(f'arff-datasets/{path}')

    def load_data(self, sample_skip=None):
        data = []
        for i, row in enumerate(arff.load(self.path)):
            if (sample_skip is not None) and (i == sample_skip):
                continue
            data.append(list(row))
        return data


class BreastCancer:
    def load_dataset(self) -> Dataset:
        x, y = load_breast_cancer(return_X_y=True)
        return Dataset('Breast Cancer', x, y ^ 1)


class Digits:
    def load_dataset(self) -> Dataset:
        x, y = load_digits(return_X_y=True)
        return Dataset('Digits', x, y)


class BankMarketing(TextParser):
    def __init__(self):
        super().__init__('bank-marketing/bank.data', spliter=';')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1, sample_skip=0)
        return Dataset('Bank-marketing', features, targets)


class CNAE9(TextParser):
    def __init__(self):
        super().__init__('cnae-9/CNAE-9.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=0)
        return Dataset('CNAE-9', features, targets)


class StatlogSegmentation(TextParser):
    def __init__(self):
        super().__init__('statlog-segmentation/segment.dat', spliter=None)

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1)
        return Dataset('Statlog Segmentation', features, targets)


class Parkinsons(TextParser):
    def __init__(self):
        super().__init__('parkinsons/parkinsons.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-7, feature_skip=0, sample_skip=0)
        return Dataset('Parkinsons', features, targets)


class Semeion(TextParser):
    def __init__(self):
        super().__init__('semeion/semeion.data', spliter=None)

    def load_dataset(self) -> Dataset:
        indexes = np.arange(-10, 0).tolist()
        features, targets = self.parse_file(index_target=indexes)
        return Dataset('Semeion', features, targets)


class Ecoli(TextParser):
    def __init__(self):
        super().__init__('ecoli/ecoli.data', spliter=None)

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1, feature_skip=0)
        indexes = np.isin(targets, [2, 3, 6], invert=True)
        return Dataset('Ecoli', features[indexes], LabelEncoder().fit_transform(targets[indexes]))


class CreditApproval(TextParser):
    def __init__(self):
        super().__init__('credit-approval/crx.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1)
        return Dataset('Credit Approval', features, targets)


class Balance(TextParser):
    def __init__(self):
        super().__init__('balance/balance-scale.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=0)
        return Dataset('Balance', features, targets)


class Zoo(TextParser):
    def __init__(self):
        super().__init__('zoo/zoo.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1, feature_skip=0)
        indexes = np.isin(targets, [2, 4], invert=True)
        return Dataset('Zoo', features[indexes], LabelEncoder().fit_transform(targets[indexes]))


class Banknote(ArffParser):
    def __init__(self):
        super().__init__('banknote-authentication.arff')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1)
        return Dataset('Banknote', features, targets)


class CarEvaluation(ArffParser):
    def __init__(self):
        super().__init__('car-evaluation.arff')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1)
        return Dataset('CarEvaluation', features, targets)


class Wilt(ArffParser):
    def __init__(self):
        super().__init__('wilt.arff')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_file(index_target=-1)
        return Dataset('Wilt', features, targets)


def get_datasets(*args) -> List[Dataset]:
    result = []
    for dataset in args:
        instance = dataset()
        result.append(instance.load_dataset())
    return result


if __name__ == '__main__':

    path = Path(__file__).parent

    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/1RHWkpKcmMWWcg'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open(path / 'datasets.zip', 'wb') as f:
        f.write(download_response.content)

    with zipfile.ZipFile(path / 'datasets.zip') as f:
        f.extractall(path=path)

    os.remove(path / 'datasets.zip')
