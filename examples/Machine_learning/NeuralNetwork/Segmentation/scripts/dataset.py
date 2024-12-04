import torch
import numpy as np

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, device, paths, signals, masks=None, common_mask=False, for_train=False):

        self._device = device
        self._paths = paths
        self._signals = [torch.Tensor(x).to(device) for x in signals]
        self._masks = [torch.LongTensor(x).to(device) for x in masks]

        self.begin_noise, self.end_noise = 1e-3, 3e-3
        self.begin_ampl, self.end_ampl = 0, 0.3

        self.begin_freq, self.end_freq = 0, 0.009

        self.prob_isoline = 0.7
        self.prob_reverse = 0.5
        self.sub_len = 4000

        self.common_mask = common_mask
        self.for_train = for_train

    def reverse_ecg(self, signal):
        result = torch.zeros_like(signal, device=self._device)
        for i, x in enumerate(signal):
            sign = 2 * (np.random.rand() < self.prob_reverse) - 1
            result[i] = sign * x
        return result

    def __len__(self):
        return len(self._signals)

    def __getitem__(self, i):
        if not self.for_train:
            return self._paths[i], self._signals[i], self.skip_borders(self._masks[i][0])

        shift = np.random.randint(0, 5000 - self.sub_len - 1)
        noise = self.begin_noise + (self.end_noise - self.begin_noise) * np.random.rand()
        signal = self._signals[i][:, shift:shift + self.sub_len] + torch.normal(0, noise,
                                                                                size=(self.sub_len,),
                                                                                device=self._device)

        signal = self.reverse_ecg(signal)

        if self._masks is None:
            return self._paths[i], signal

        mask = self._masks[i][:, shift: shift + self.sub_len]
        indexes = torch.randperm(12, device=self._device)

        if self.common_mask:
            mask = mask[0]
        else:
            mask = mask[indexes]

        return self._paths[i], signal[indexes], self.skip_borders(mask)

    def skip_borders(self, mask):
        wave_start = torch.logical_and(torch.roll(mask, 1) == 0, mask != 0).type(torch.uint8)
        wave_finish = torch.logical_and(torch.roll(mask, -1) == 0, mask != 0).type(torch.uint8)

        indexes_starts, = torch.where(wave_start == 1)
        indexes_finish, = torch.where(wave_finish == 1)

        left_skip = indexes_starts[indexes_starts > 500][0]
        right_skip = indexes_finish[indexes_finish < len(mask) - 500][-1]

        mask_copy = torch.clone(mask)
        mask_copy[:left_skip] = 4
        mask_copy[right_skip:] = 4

        return mask_copy