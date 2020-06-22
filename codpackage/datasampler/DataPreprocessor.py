from .TrainCODDataset import TrainCODDataset
from .TestCODDataset import TestCODDataset

from torch.utils.data import DataLoader
import numpy as np

import random

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.train_dataset = TrainCODDataset(self.config.TRAIN.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)
        self.val_dataset = TestCODDataset(self.config.VAL.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)

        self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size = self.config.TRAIN.BATCH_SIZE,
                                            num_workers = self.config.TRAIN.WORKERS,
                                            pin_memory = True,
                                            shuffle = self.config.TRAIN.SHUFFLE,
                                            worker_init_fn = lambda wid: random.seed(self.seed + wid))
        self.val_dataloader = DataLoader(self.val_dataset,
                                        batch_size = self.config.TRAIN.BATCH_SIZE,
                                        num_workers = self.config.TRAIN.WORKERS,
                                        pin_memory = True,
                                        shuffle = self.config.TRAIN.SHUFFLE,
                                        worker_init_fn = lambda wid: random.seed(self.seed + wid))
        # if self.test_dataset is not None:
        #     self.test_dataloader = DataLoader(self.test_dataset,
        #                                     batch_size = self.config['batch_size'],
        #                                     num_workers = 8,
        #                                     pin_memory = True,
        #                                     shuffle = True,
        #                                     worker_init_fn = lambda wid: random.seed(self.seed + wid))

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_val_dataloader(self):
        return self.val_dataloader