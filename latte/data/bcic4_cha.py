from typing import Optional
import os

import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule, make_collate_fn

# Maps 1-based index (1..16) -> actual BCIcha subject file number
_SUBJECT_FILE_IDS = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


class BCIcha(BaseDataModule):
    all_subject_ids = list(range(1, 17))   # 1-based indices into _SUBJECT_FILE_IDS
    class_names = ["class_0", "class_1"]
    channels = 56
    classes = 2
    num_subjects = 16

    def __init__(self, preprocessing_dict, subject_id, include_subject_ids=False,
                 cross_subject_normalize=False):
        super().__init__(preprocessing_dict, subject_id)
        self.include_subject_ids = include_subject_ids
        self.cross_subject_normalize = cross_subject_normalize
        self.data_path = preprocessing_dict.get("data_path", "./data/BCIcha/")
        self.val_dataset = None

    def prepare_data(self) -> None:
        file_num = _SUBJECT_FILE_IDS[int(self.subject_id) - 1]
        path = os.path.join(self.data_path, f'Data_S{file_num:02d}_Sess.mat')
        data = io.loadmat(path)
        self._raw_x = data['x_test']        # (340, 56, 160)
        self._raw_y = data['y_test'].ravel() # (340,)

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, '_raw_x'):
            self.prepare_data()

        X, y = self._raw_x, self._raw_y
        X_train, y_train = X[:180],     y[:180]
        X_val,   y_val   = X[180:240],  y[180:240]
        X_test,  y_test  = X[240:340],  y[240:340]

        if self.preprocessing_dict.get("z_scale", True):
            if self.cross_subject_normalize:
                # fit on all subjects' pooled training data
                X_all_tr = np.concatenate([
                    io.loadmat(os.path.join(self.data_path, f'Data_S{fn:02d}_Sess.mat'))['x_test'][:180]
                    for fn in _SUBJECT_FILE_IDS
                ])
                _, c, t = X_all_tr.shape
                sc = StandardScaler().fit(X_all_tr.transpose(1, 0, 2).reshape(c, -1).T)

                def _apply(arr):
                    s = arr.shape[0]
                    return sc.transform(arr.transpose(1, 0, 2).reshape(c, -1).T).T.reshape(c, s, t).transpose(1, 0, 2)

                X_train = _apply(X_train)
                X_val   = _apply(X_val)
                X_test  = _apply(X_test)
            else:
                X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        sid = self.subject_id
        train_sids = np.full(len(y_train), sid) if self.include_subject_ids else None
        val_sids   = np.full(len(y_val),   sid) if self.include_subject_ids else None
        test_sids  = np.full(len(y_test),  sid) if self.include_subject_ids else None

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, train_sids)
        self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val,   y_val,   val_sids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test,  y_test,  test_sids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
        )


class BCIchaLOSO(BaseDataModule):
    """Leave-one-subject-out: train on 15 subjects, test on the held-out one."""
    all_subject_ids = list(range(1, 17))
    class_names = ["class_0", "class_1"]
    channels = 56
    classes = 2
    num_subjects = 16

    def __init__(self, preprocessing_dict, subject_id, include_subject_ids=False):
        super().__init__(preprocessing_dict, subject_id)
        self.include_subject_ids = include_subject_ids
        self.data_path = preprocessing_dict.get("data_path", "./data/BCIcha/")
        self.val_dataset = None

    def prepare_data(self) -> None:
        pass

    def _load_subject(self, sid):
        file_num = _SUBJECT_FILE_IDS[int(sid) - 1]
        path = os.path.join(self.data_path, f'Data_S{file_num:02d}_Sess.mat')
        data = io.loadmat(path)
        return data['x_test'], data['y_test'].ravel()

    def setup(self, stage: Optional[str] = None) -> None:
        train_subjects = [s for s in self.all_subject_ids if s != int(self.subject_id)]

        X_tr, y_tr, sids_tr = [], [], []
        X_va, y_va, sids_va = [], [], []
        for sid in train_subjects:
            X, y = self._load_subject(sid)
            X_tr.append(X[:180]);    y_tr.append(y[:180]);    sids_tr.append(np.full(180, sid))
            X_va.append(X[180:240]); y_va.append(y[180:240]); sids_va.append(np.full(60,  sid))

        X_train = np.concatenate(X_tr); y_train = np.concatenate(y_tr)
        X_val   = np.concatenate(X_va); y_val   = np.concatenate(y_va)

        X_test_raw, y_test_raw = self._load_subject(self.subject_id)
        X_test, y_test = X_test_raw[240:340], y_test_raw[240:340]

        if self.preprocessing_dict.get("z_scale", True):
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        train_sids = np.concatenate(sids_tr) if self.include_subject_ids else None
        val_sids   = np.concatenate(sids_va) if self.include_subject_ids else None
        test_sids  = np.full(len(y_test), self.subject_id) if self.include_subject_ids else None

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, train_sids)
        self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val,   y_val,   val_sids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test,  y_test,  test_sids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
            collate_fn=make_collate_fn(self.preprocessing_dict, training=False),
        )


class BCIchaAll(BaseDataModule):
    """All 16 BCIcha subjects for joint cross-subject training with subject-ID LoRA."""
    all_subject_ids = list(range(1, 17))
    class_names = ["class_0", "class_1"]
    channels = 56
    classes = 2
    num_subjects = 16

    def __init__(self, preprocessing_dict, subject_id="all", include_subject_ids=True):
        super().__init__(preprocessing_dict, subject_id=1)
        self.include_subject_ids = include_subject_ids
        self.data_path = preprocessing_dict.get("data_path", "./data/BCIcha/")
        self.val_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        X_tr, y_tr, sids_tr = [], [], []
        X_va, y_va, sids_va = [], [], []
        X_te, y_te, sids_te = [], [], []

        for idx, file_num in enumerate(_SUBJECT_FILE_IDS):
            sid = idx + 1  # 1-based subject index for LoRA
            path = os.path.join(self.data_path, f'Data_S{file_num:02d}_Sess.mat')
            data = io.loadmat(path)
            X = data['x_test']        # (340, 56, 160)
            y = data['y_test'].ravel()

            X_tr.append(X[:180]);    y_tr.append(y[:180]);    sids_tr.append(np.full(180, sid))
            X_va.append(X[180:240]); y_va.append(y[180:240]); sids_va.append(np.full(60,  sid))
            X_te.append(X[240:340]); y_te.append(y[240:340]); sids_te.append(np.full(100, sid))

        X_train = np.concatenate(X_tr); y_train = np.concatenate(y_tr)
        X_val   = np.concatenate(X_va); y_val   = np.concatenate(y_va)
        X_test  = np.concatenate(X_te); y_test  = np.concatenate(y_te)

        if self.preprocessing_dict.get("z_scale", True):
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        train_sids = np.concatenate(sids_tr) if self.include_subject_ids else None
        val_sids   = np.concatenate(sids_va) if self.include_subject_ids else None
        test_sids  = np.concatenate(sids_te) if self.include_subject_ids else None

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, train_sids)
        self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val,   y_val,   val_sids)
        self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test,  y_test,  test_sids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
            collate_fn=make_collate_fn(self.preprocessing_dict, training=False),
        )
