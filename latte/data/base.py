from typing import Dict, Optional

import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import os

from latte.data.interaug import interaug
def make_collate_fn(preproc, training=True):
    """Return a collate function that handles 2-tuple (x, y) or 3-tuple (x, sub, y) batches."""
    def collate(batch):
        if len(batch[0]) == 3:
            xs, subs, ys = zip(*batch)
            x = torch.stack(xs)
            sub = torch.stack(list(subs))
            y = torch.stack(list(ys))
            if training and preproc.get("interaug", False):
                x, y, sub = interaug([x, y], subject_ids=sub)
            return x, sub, y
        else:
            xs, ys = zip(*batch)
            x = torch.stack(xs)
            y = torch.tensor(ys, dtype=torch.long)
            if training and preproc.get("interaug", False):
                x, y = interaug([x, y])
            return x, y
    return collate


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          shuffle=True,
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                          persistent_workers=True,          # keeps workers alive between epochs
                          prefetch_factor=4,                 # each worker preloads 4 future batches
                          collate_fn=make_collate_fn(self.preprocessing_dict)
                    )

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                          persistent_workers=True,          # keeps workers alive between epochs
                          prefetch_factor=4,                 # each worker preloads 4 future batches
                        )

    @staticmethod
    # per-channel & per-timepoint across samples
    # def _z_scale(X, X_test):
    #     for ch_idx in range(X.shape[1]):
    #         sc = StandardScaler()
    #         X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
    #         X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
    #     return X, X_test
    # Per-channel across all samples and timepoints
    def _z_scale(X, X_test):
        # reshape to (samples*time, channels)
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_test

    # @staticmethod
    # # per-channel & per-timepoint across samples
    # def _z_scale_tvt(X_train, X_val, X_test):
    #     for ch in range(X_train.shape[1]):
    #         sc = StandardScaler()
    #         X_train[:, ch, :] = sc.fit_transform(X_train[:, ch, :])
    #         X_val[:, ch, :] = sc.transform(X_val[:, ch, :])
    #         X_test[:, ch, :] = sc.transform(X_test[:, ch, :])
    #     return X_train, X_val, X_test

    # Per-channel across all samples and timepoints
    def _z_scale_tvt(X, X_val, X_test):
        # reshape to (samples*time, channels)
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_val_2d = X_val.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_val = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_val, X_test
    
    @staticmethod
    def _make_tensor_dataset(X, y, subject_ids=None):
        """Create a TensorDataset, optionally including subject IDs.
        
        Args:
            X: Data array
            y: Labels array  
            subject_ids: Optional array of subject IDs per sample
            
        Returns:
            TensorDataset yielding (X, y) or (X, subject_id, y) tuples
        """
        X_tensor = torch.Tensor(X)
        y_tensor = torch.Tensor(y).type(torch.LongTensor)
        
        if subject_ids is not None:
            subject_tensor = torch.Tensor(subject_ids).type(torch.LongTensor)
            return TensorDataset(X_tensor, subject_tensor, y_tensor)
        return TensorDataset(X_tensor, y_tensor)
        
    # @staticmethod
    # def _make_tensor_dataset(X, y, preprocessing_dict=None, mode="train"):
    #     if preprocessing_dict and mode == "train":
    #         return AugmentedTensorDataset(
    #             X, y,
    #             interaug=preprocessing_dict.get("interaug", False),
    #         )
    #     return TensorDataset(torch.tensor(X), torch.tensor(y).long())


# from latte.data.interaug import interaug
# class AugmentedTensorDataset(TensorDataset):
#     def __init__(self, X, y, interaug=False):
#         super().__init__(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
#         self.interaug = interaug

#     def __getitem__(self, index):
#         x, y = super().__getitem__(index)
        
#         if self.interaug:
#             x, y = interaug([x, y])

#         return x, y
