import random

from examples.Machine_learning.NeuralNetwork.Segmentation.scripts.dataset import SegmentationDataset
from examples.Machine_learning.NeuralNetwork.Segmentation.scripts.metric import AllMetricTracker
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from typing import Dict
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import numpy as np
from lightning.pytorch import LightningModule
from examples.Machine_learning.NeuralNetwork.Segmentation.scripts.metric import SegmentationMetric
from examples.Machine_learning.NeuralNetwork.Segmentation.scripts.model import Encoder, Decoder, UNet


class UnetModule(LightningModule):
    def __init__(self, kernel_size=23, q=1.2, label_smoothing=0, p=0.75):
        super().__init__()
        self.save_hyperparameters()
        encoder = Encoder(12, kernel_size=kernel_size, q=q, p=p)
        decoder = Decoder(encoder, 4)

        self.model = UNet(encoder, decoder)
        self.loss = nn.CrossEntropyLoss(ignore_index=4, label_smoothing=label_smoothing)

        self.p_metric = SegmentationMetric('p', 'all', return_type='f1', samples=150)
        self.t_metric = SegmentationMetric('t', 'all', return_type='f1', samples=150)
        self.qrs_metric = SegmentationMetric('qrs', 'all', return_type='f1', samples=150)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            x = x.unsqueeze(0) if len(x.shape) == 2 else x
        x = x.to(self.device)
        logits = self.model(x)
        y_pred = logits.argmax(axis=1)
        return y_pred.cpu().detach().numpy()

    def training_step(self, batch):
        _, x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        dict_ = {'train_loss': loss}
        self.log_dict(dict_, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch):
        _, x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        dict_ = {'val_loss': loss}

        metrics = self.get_metric(x, y, 'val')
        dict_.update(metrics)

        self.log_dict(dict_, on_epoch=True, on_step=False)

        return loss

    def get_metric(self, x, y_true, prefix):
        y_true = y_true.cpu().detach().numpy()
        y_pred = self.predict(x)
        p_f1_score = self.p_metric(y_pred, y_true)
        qrs_f1_score = self.qrs_metric(y_pred, y_true)
        t_f1_score = self.t_metric(y_pred, y_true)
        dict = {f'{prefix}_p_wave': p_f1_score, f'{prefix}_qrs_wave': qrs_f1_score, f'{prefix}_t_wave': t_f1_score}
        return dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=50)
        return [optimizer], [{"scheduler": scheduler,
                              "interval": "epoch",
                              "monitor": "train_loss"}]

def get_dataset(paths):
    return [np.load(f'data/signals/{x}') for x in paths], \
           [np.load(f'data/masks/{x}') for x in paths]

class Cardio2D(Problem):
    def __init__(self, p_bound: Dict[str, float], q_bound: Dict[str, float]):
        super(Cardio2D, self).__init__()
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        ecg_list = sorted(os.listdir('data/signals/'))
        ecg_list = [x for x in ecg_list if x.split('_')[-1] != 'unsupervised.npy']

        train_list, test_list = train_test_split(ecg_list, test_size=0.2, shuffle=True, random_state=42)

        for x in sorted(os.listdir('data/signals/')):
            if x.split('_')[-1] == 'unsupervised.npy':
                train_list.append(x)

        x_train, y_train = get_dataset(train_list)
        x_test, y_test = get_dataset(test_list)

        train_dataset = SegmentationDataset('cpu', train_list, x_train, y_train, common_mask=True, for_train=True)
        val_dataset = SegmentationDataset('cpu', test_list, x_test, y_test, common_mask=True)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        self.float_variable_names = np.array(["P parameter", "Q parameter"], dtype=str)
        self.lower_bound_of_float_variables = np.array([p_bound['low'], q_bound['low']],
                                                       dtype=np.double)
        self.upper_bound_of_float_variables = np.array([p_bound['up'], q_bound['up']],
                                                       dtype=np.double)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        p, q = point.float_variables[0], point.float_variables[1]

        now = datetime.now().strftime('%d.%m.%Y_%H:%M:%S')

        checkpoint = ModelCheckpoint(dirpath=f'models/',
                                     filename=f"{random.uniform(1, 100):.9f}" + " " + f"{p:.9f}" + '_' + f"{q:.9f}" + '_' + '{epoch}_{val_p_wave:.6f}_{val_qrs_wave:.6f}_{val_t_wave:.6f}',
                                     monitor='val_p_wave',
                                     save_top_k=3,
                                     mode='max')
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=300)

        cb = AllMetricTracker()
        model = UnetModule(p=p, q=q)
        trainer = Trainer(max_epochs=1_000_000, callbacks=[checkpoint, early_stopping, cb])
        try:
            trainer.fit(model, self.train_loader, self.val_loader)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

        print('p ' + f"{p:.9f}")
        print('q ' + f"{q:.9f}")
        function_value.value = -cb.best_p_valscore
        print(-cb.best_p_valscore)
        return function_value