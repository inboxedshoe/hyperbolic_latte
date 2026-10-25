from typing import Optional
import os

import numpy as np
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule, make_collate_fn
from .bcic4_2a import _extract_data_from_dataset
from latte.data.load_bcic4 import load_bcic4


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2
    num_subjects = 9

    def __init__(self, preprocessing_dict, subject_id, include_subject_ids=False):
        super().__init__(preprocessing_dict, subject_id)
        self.include_subject_ids = include_subject_ids

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2b",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        splitted_ds = self.dataset.split("session")
        train_datasets = [splitted_ds[f"{s}train"] for s in [0, 1, 2]]
        test_datasets  = [splitted_ds[f"{s}test"]  for s in [3, 4]]

        X = np.concatenate([_extract_data_from_dataset(ds)[0] for ds in train_datasets], axis=0)
        y = np.concatenate([_extract_data_from_dataset(ds)[1] for ds in train_datasets], axis=0)
        X_test = np.concatenate([_extract_data_from_dataset(ds)[0] for ds in test_datasets], axis=0)
        y_test = np.concatenate([_extract_data_from_dataset(ds)[1] for ds in test_datasets], axis=0)

        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        sid = self.subject_id
        train_sids = np.full(len(y), sid) if self.include_subject_ids else None
        test_sids = np.full(len(y_test), sid) if self.include_subject_ids else None

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, train_sids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, test_sids)


class BCICIV2bLOSO(BCICIV2b):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int, include_subject_ids=False):
        super().__init__(preprocessing_dict, subject_id, include_subject_ids)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        splitted_ds = self.dataset.split("subject")
        train_subjects = [s for s in self.all_subject_ids if s != self.subject_id]

        X_tr, y_tr, sids_tr = [], [], []
        X_va, y_va, sids_va = [], [], []
        for s in train_subjects:
            sess = splitted_ds[str(s)].split("session")
            for k in [0, 1, 2]:
                X_part, y_part = _extract_data_from_dataset(sess[f"{k}train"])
                X_tr.append(X_part); y_tr.append(y_part)
                sids_tr.append(np.full(len(y_part), s))
            for k in [3, 4]:
                X_part, y_part = _extract_data_from_dataset(sess[f"{k}test"])
                X_va.append(X_part); y_va.append(y_part)
                sids_va.append(np.full(len(y_part), s))

        test_sess = splitted_ds[str(self.subject_id)].split("session")
        X_te, y_te = [], []
        for k in [3, 4]:
            X_part, y_part = _extract_data_from_dataset(test_sess[f"{k}test"])
            X_te.append(X_part); y_te.append(y_part)

        X = np.concatenate(X_tr, axis=0)
        y = np.concatenate(y_tr, axis=0)
        X_val = np.concatenate(X_va, axis=0)
        y_val = np.concatenate(y_va, axis=0)
        X_test = np.concatenate(X_te, axis=0)
        y_test = np.concatenate(y_te, axis=0)

        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        train_sids = np.concatenate(sids_tr) if self.include_subject_ids else None
        val_sids = np.concatenate(sids_va) if self.include_subject_ids else None
        test_sids = np.full(len(y_test), self.subject_id) if self.include_subject_ids else None

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, train_sids)
        self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, val_sids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, test_sids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                          collate_fn=make_collate_fn(self.preprocessing_dict, training=False),
                          )


class BCICIV2bAll(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2
    num_subjects = 9

    def __init__(self, preprocessing_dict, subject_id="all", include_subject_ids=True):
        super().__init__(preprocessing_dict, subject_id=1)
        self.include_subject_ids = include_subject_ids

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage=None) -> None:
        if self.dataset is None:
            self.prepare_data()
        splitted_ds = self.dataset.split("subject")
        X_train_list, y_train_list, subj_train_list = [], [], []
        X_test_list,  y_test_list,  subj_test_list  = [], [], []
        for subj_id in self.all_subject_ids:
            sess = splitted_ds[str(subj_id)].split("session")
            for s in [0, 1, 2]:
                X_part, y_part = _extract_data_from_dataset(sess[f"{s}train"])
                X_train_list.append(X_part)
                y_train_list.append(y_part)
                subj_train_list.append(np.full(len(y_part), subj_id))
            for s in [3, 4]:
                X_part, y_part = _extract_data_from_dataset(sess[f"{s}test"])
                X_test_list.append(X_part)
                y_test_list.append(y_part)
                subj_test_list.append(np.full(len(y_part), subj_id))
        X      = np.concatenate(X_train_list, axis=0)
        y      = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list,  axis=0)
        y_test = np.concatenate(y_test_list,  axis=0)
        train_subject_ids = np.concatenate(subj_train_list, axis=0) if self.include_subject_ids else None
        test_subject_ids  = np.concatenate(subj_test_list,  axis=0) if self.include_subject_ids else None
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, train_subject_ids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, test_subject_ids)
