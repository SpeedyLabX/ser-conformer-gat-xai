"""Data module placeholder: loaders and datamodule for IEMOCAP / RAVDESS / MELD"""
from typing import Optional

class DataModule:
    def __init__(self, root: str, batch_size: int = 16, num_workers: int = 4):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # TODO: implement dataset loading and splits
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
