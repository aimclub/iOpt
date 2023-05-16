import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder


class Dataset(ABC):
    def __init__(self, path):
        self.path = path

    @abstractmethod
    def load_features(self):
        pass

    @abstractmethod
    def load_targets(self):
        pass

class ICUDataset(Dataset):
    def __init__(self, path, spliter, skips='?'):
        super().__init__(path)
        self.spliter = spliter
        self.skips = skips
        self.data = self.load_uci()

    def load_uci(self):
        with open(Path(__file__).parent / self.path) as f:
            data = []
            while (row := f.readline()).strip():
                input_ = [x.strip() for x in row.split(self.spliter)]
                if self.skips in input_:
                    continue
                data.append(input_)
        return data
    
    @staticmethod
    def load_features(x):
        features = len(x[0])
        return np.array(
            [ICUDataset.preprocess_feature([sample[i] for sample in x]) 
             for i in range(features)]
        ).T

    @staticmethod
    def load_targets(y):
        return np.array(ICUDataset.preprocess_feature(y))
    
    @staticmethod
    def preprocess_feature(x):
        try:
            return [float(value) for value in x]
        except ValueError:
            return LabelEncoder().fit_transform(x)

class Adult(ICUDataset):
    def __init__(self):
        super().__init__('adult/adult.data', ',')
        self.x = [x[:-1] for x in self.data]
        self.y = [x[-1] for x in self.data]
    
    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)

class Banknote(ICUDataset):
    def __init__(self):
        super().__init__('banknote/data_banknote_authentication.txt', ',')
        self.x = [x[:-1] for x in self.data]
        self.y = [x[-1] for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)
    
class BreastCancer(ICUDataset):
    def __init__(self):
        super().__init__('breast cancer/breast-cancer-wisconsin.data', ',')
        self.x = [x[1:-1] for x in self.data]
        self.y = [int(x[-1]) // 2 - 1 for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)
    
class CreditApproval(ICUDataset):
    def __init__(self):
        super().__init__('creditapproval/crx.data', ',')
        self.x = [x[:-1] for x in self.data]
        self.y = [x[-1] for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)

class Ecoli(ICUDataset):
    def __init__(self):
        super().__init__('ecoli/ecoli.data', None)
        self.x = [x[1:-1] for x in self.data]
        self.y = [x[-1] for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)
    
class Balance(ICUDataset):
    def __init__(self):
        super().__init__('balance/balance-scale.data', ',')
        self.x = [x[1:] for x in self.data]
        self.y = [x[0] for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)

class Parkinsons(ICUDataset):
    def __init__(self):
        super().__init__('parkinsons/parkinsons.data', ',')
        self.data = self.data[1:]
        self.x = [[xx for i, xx in enumerate(x) if (i != 0) and (i != 17)] for x in self.data]
        self.y = [x[17] for x in self.data]
    
    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)
    
class Zoo(ICUDataset):
    def __init__(self):
        super().__init__('zoo/zoo.data', ',')
        self.x = [x[1:-1] for x in self.data]
        self.y = [x[-1] for x in self.data]

    def load_features(self):
        return super().load_features(self.x)
    
    def load_targets(self):
        return super().load_targets(self.y)

ALL_DATASET = [Banknote, BreastCancer, CreditApproval, Ecoli, Balance, Parkinsons, Zoo]


def load_all_dataset():
    result = []
    for dataset in ALL_DATASET:
        instance = dataset()
        result.append((dataset.__name__, instance.load_features(), instance.load_targets()))
    return result