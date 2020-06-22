from .CODDataset import CODDataset

class TrainCODDataset(CODDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)