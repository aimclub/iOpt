from typing import Literal
import numpy as np
import math
from lightning.pytorch.callbacks import Callback

class AllMetricTracker(Callback):
  def __init__(self):
    self.collection = []
    self.best_p_valscore = -1
    self.best_qrs_valscore = -1
    self.best_t_valscore = -1
    self.l2_max = 0

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.collection.append(elogs)
    if self.best_p_valscore < elogs['val_p_wave'].item():
        self.best_p_valscore = elogs['val_p_wave'].item()
        self.best_qrs_valscore = elogs['val_qrs_wave'].item()
        self.best_t_valscore = elogs['val_t_wave'].item()
    cur_l2 = math.sqrt(elogs['val_p_wave'].item() * elogs['val_p_wave'].item() + elogs['val_qrs_wave'].item() * elogs['val_qrs_wave'].item() + elogs['val_t_wave'].item() * elogs['val_t_wave'].item())
    self.l2_max = max(cur_l2, self.l2_max)
    print(f'P_VAL: {self.best_p_valscore:.6f}, QRS_VAL: {self.best_qrs_valscore:.6f}, T_VAL: {self.best_t_valscore:.6f}, L_VEC: {self.l2_max:.6f}')


class SegmentationMetric:
    def __init__(self,
                 monitor: Literal['p', 'qrs', 't', 'all'] = 'all',
                 orientation_type: Literal['onset', 'offset', 'all'] = 'all',
                 return_type: Literal['precision', 'recall', 'f1', 'confusion_matrix'] = 'confusion_matrix',
                 samples=75):

        assert monitor in ['p', 'qrs', 't', 'all']
        assert orientation_type in ['onset', 'offset', 'all']
        assert return_type in ['precision', 'recall', 'f1', 'confusion_matrix']

        self.samples = samples
        self.monitor = monitor
        self.orientation_type = orientation_type
        self.return_type = return_type

        self.metric_to_func = {'precision': self.__precision,
                               'recall': self.__recall,
                               'f1': self.__f1}

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        assert len(y_pred.shape) == 2

        matrix = np.zeros((2, 2), dtype=int)
        monitors = ['p', 'qrs', 't'] if self.monitor == 'all' else [self.monitor]
        orientations = ['onset', 'offset'] if self.orientation_type == 'all' else [self.orientation_type]
        for wave in monitors:
            for orientation in orientations:
                matrix += self.__handle(y_pred, y_true, wave, orientation)

        if self.return_type == 'confusion_matrix':
            return matrix

        return self.metric_to_func[self.return_type](matrix[0, 1], matrix[1, 0], matrix[1, 1])

    def __handle(self, y_pred, y_true, wave, orientation) -> tuple[int, int, int]:

        index = ['p', 'qrs', 't'].index(wave) + 1
        orientation = 2 * ['offset', 'onset'].index(orientation) - 1
        y_pred[y_true == 4] = 0

        y_true, y_pred = (y_true == index), (y_pred == index)

        wave_true = np.logical_and(np.roll(y_true, orientation) != 1, y_true == 1).astype(int)
        wave_pred = np.logical_and(np.roll(y_pred, orientation) != 1, y_pred == 1).astype(int)

        true_batch, true_indexes = np.where(wave_true == 1)

        tp = fn = 0

        for batch, x in zip(true_batch, true_indexes):
            wave = wave_pred[batch][x - self.samples // 2: x + self.samples // 2]
            if wave.sum():
                tp += 1
            else:
                fn += 1
            wave[:] = -1

        fp = (wave_pred[:, self.samples:-self.samples] == 1).sum()
        return np.array([[0, fp], [fn, tp]])

    @staticmethod
    def __precision(fp, fn, tp):
        if fp + tp == 0:
            return 1
        return tp / (tp + fp)

    @staticmethod
    def __recall(fp, fn, tp):
        if fn + tp == 0:
            return 1
        return tp / (tp + fn)

    @staticmethod
    def __f1(fp, fn, tp):
        precision = SegmentationMetric.__precision(fp, fn, tp)
        recall = SegmentationMetric.__recall(fp, fn, tp)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def __str__(self):
        return f'{self.monitor}_{self.orientation_type}'