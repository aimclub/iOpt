import pandas as pd
import numpy as np

from hyperparams import Hyperparameter
from frameworks import Searcher, Point
from data.loader import Parser, get_datasets
from metrics import DATASET_TO_METRIC
from metrics import Metric

from multiprocessing import Pool
from itertools import product
from collections import defaultdict
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


class Experiment:
    def __init__(self, estimator,
                 hyperparams: dict[str, Hyperparameter],
                 searchers: list[Searcher],
                 parsers: list[Parser],
                 metric: Metric = None):

        self.estimator = estimator
        self.hyperparams = hyperparams
        self.searchers = {str(x): x for x in searchers}

        datasets = get_datasets(*parsers)

        self.datasets = {x.name: x for x in datasets}
        self.metrics = {x.name: DATASET_TO_METRIC[y] if (metric is None) else metric
                        for x, y in zip(datasets, parsers)}

    @ignore_warnings(category=ConvergenceWarning)
    def run(self, dir, non_deterministic_trials: int = 1, n_jobs: int = 1) -> pd.DataFrame:

        assert non_deterministic_trials > 0, 'Something very strange'
        assert n_jobs >= -1, 'Something very strange'
        self.non_deterministic_trials = non_deterministic_trials
        self.n_jobs = n_jobs

        frame = self.start_pool()

        values, times = frame.applymap(self.points_to_values), \
            frame.applymap(self.points_to_times)

        Path(dir).mkdir(parents=True, exist_ok=True)
        values.to_csv(f'{dir}/metrics.csv')
        times.to_csv(f'{dir}/times.csv')

    def start_pool(self):
        trials = self.get_trials()
        frame = defaultdict(lambda: defaultdict(list))

        with Pool(self.n_jobs) as pool:
            result = pool.starmap(self.objective, trials)

            for dataset, searcher, points in result:
                frame[searcher][dataset].append(points)

        return pd.DataFrame(frame)

    def get_trials(self):
        names = []
        for _, (name, searcher) in enumerate(self.searchers.items(), start=1):
            value = 1 if searcher.is_deterministic else self.non_deterministic_trials
            names.extend((name for _ in range(1, value + 1)))
        return list((x, y) for x, y in product(list(self.datasets), names))

    def objective(self, dname: str, sname: str):
        searcher, dataset, metric = self.searchers[sname], \
                                    self.datasets[dname], \
                                    self.metrics[dname]
        points = searcher.tune(self.estimator, self.hyperparams, dataset, metric)
        return dname, sname, points

    @staticmethod
    def points_to_values(lists_of_points: list[list[Point]]):
        best_values = [max(x.value for x in point) for point in lists_of_points]
        if len(best_values) == 1:
            return f'{best_values[0]:.3f}'
        mean, std = np.mean(best_values), np.std(best_values)
        return f'{mean:.3f} ± {Experiment.percentile(mean, std):.2f}%'

    @staticmethod
    def points_to_times(lists_of_points: list[list[Point]]):
        times = [(point[-1].timepoint - point[0].timepoint) / 1000 for point in lists_of_points]
        if len(times) == 1:
            return Experiment.seconds_to_str(times[0])
        mean, std = np.mean(times), np.std(times)
        return f'{Experiment.seconds_to_str(mean)} ± {Experiment.percentile(mean, std):.2f}%'

    @staticmethod
    def percentile(mean, std):
        return std / mean * 100

    @staticmethod
    def seconds_to_str(s):
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
