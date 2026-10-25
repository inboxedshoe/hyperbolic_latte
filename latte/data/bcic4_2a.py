
from typing import Optional
import numpy as np
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from latte.data.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


def _get_session_keys(splitted_ds):
    """
    Get the train and test session keys from a split dataset.
    Handles different naming conventions across MOABB/braindecode versions:
    - Old: 'session_T', 'session_E'
    - New: '0train', '1test' or 'train', 'test' or '0', '1'
    """
    keys = list(splitted_ds.keys())
    
    # Try old naming convention first
    if "session_T" in keys and "session_E" in keys:
        return "session_T", "session_E"
    
    # Try new naming conventions
    train_key = None
    test_key = None
    
    for key in keys:
        key_lower = str(key).lower()
        if "train" in key_lower or key_lower in ["0", "t"]:
            train_key = key
        elif "test" in key_lower or "eval" in key_lower or key_lower in ["1", "e"]:
            test_key = key
    
    if train_key is not None and test_key is not None:
        return train_key, test_key
    
    # Fallback: assume first key is train, second is test
    if len(keys) >= 2:
        return keys[0], keys[1]
    
    raise KeyError(f"Could not determine train/test session keys from: {keys}")


def _extract_data_from_dataset(dataset, return_subject_ids=False):
    """
    Extract X (data) and y (labels) from a braindecode WindowsDataset.
    Handles both old and new braindecode APIs:
    - Old: run.windows.load_data()._data and run.y
    - New: iterate through dataset items as (X, y, idx) tuples
    
    Args:
        dataset: A braindecode WindowsDataset
        return_subject_ids: If True, also return an array of subject IDs
            corresponding to each sample
    
    Returns:
        X, y: Data and labels arrays
        X, y, subject_ids: If return_subject_ids=True, also returns subject IDs
    """
    X_list = []
    y_list = []
    subject_ids_list = []
    
    for run in dataset.datasets:
        # Get subject ID from dataset description if needed
        if return_subject_ids:
            # Extract subject ID from the dataset's description
            if hasattr(run, 'description') and 'subject' in run.description:
                subj_id = run.description['subject']
            elif hasattr(run, 'description'):
                # Try to get from any available metadata
                subj_id = run.description.get('subject', -1)
            else:
                subj_id = -1
        
        # Try old API first
        try:
            if hasattr(run, 'windows'):
                run_data = run.windows.load_data()._data
                run_labels = np.array(run.y)
            else:
                # New API: run is a WindowsDataset, iterate through it
                raise AttributeError("No windows attribute, try new API")
        except (AttributeError, TypeError):
            # New braindecode API: dataset items are (X, y, idx) or (X, y) tuples
            run_X = []
            run_y = []
            for item in run:
                if len(item) == 3:  # (X, y, idx)
                    x, y, _ = item
                else:  # (X, y)
                    x, y = item
                # Convert tensor to numpy if needed
                if hasattr(x, 'numpy'):
                    x = x.numpy()
                run_X.append(x)
                run_y.append(y)
            run_data = np.array(run_X)
            run_labels = np.array(run_y)
        
        X_list.append(run_data)
        y_list.append(run_labels)
        
        if return_subject_ids:
            # Create array of subject IDs with same length as this run's data
            subject_ids_list.append(np.full(len(run_data), subj_id))
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if return_subject_ids:
        subject_ids = np.concatenate(subject_ids_list, axis=0)
        return X, y, subject_ids
    return X, y


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4
    num_subjects = 9
    
    val_dataset = None

    def __init__(self, preprocessing_dict, subject_id, include_subject_ids=False, val_split=None):
        super().__init__(preprocessing_dict, subject_id)
        self.include_subject_ids = include_subject_ids
        self.val_split = val_split

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        train_key, test_key = _get_session_keys(splitted_ds)
        train_dataset, test_dataset = splitted_ds[train_key], splitted_ds[test_key]

        # load the data
        X, y = _extract_data_from_dataset(train_dataset)
        X_test, y_test = _extract_data_from_dataset(test_dataset)

        if self.val_split is not None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.val_split,
                random_state=self.preprocessing_dict.get("seed", 42), stratify=y)
            if self.preprocessing_dict["z_scale"]:
                X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)
            val_subject_ids = np.full(len(y_val), self.subject_id) if self.include_subject_ids else None
            self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val, val_subject_ids)
        else:
            if self.preprocessing_dict["z_scale"]:
                X, X_test = BaseDataModule._z_scale(X, X_test)

        # Create subject ID arrays if needed
        train_subject_ids = np.full(len(y), self.subject_id) if self.include_subject_ids else None
        test_subject_ids = np.full(len(y_test), self.subject_id) if self.include_subject_ids else None

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, train_subject_ids)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, test_subject_ids)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.preprocessing_dict["batch_size"],
                              num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                              pin_memory=True,
                              )
        return self.test_dataloader()


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4
    num_subjects = 9

    def __init__(self, preprocessing_dict, subject_id, include_subject_ids=False):
        super().__init__(preprocessing_dict, subject_id)
        self.include_subject_ids = include_subject_ids

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by session
        splitted_ds = self.dataset.split("session")
        train_key, test_key = _get_session_keys(splitted_ds)
        session1 = splitted_ds[train_key]  # training + validation
        session2 = splitted_ds[test_key]  # testing only
        
        # Load session 1 data
        X, y = _extract_data_from_dataset(session1)

        # Split session 1: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        # Load session 2 as test set
        X_test, y_test = _extract_data_from_dataset(session2)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        # Create subject ID arrays if needed
        train_subject_ids = np.full(len(y_train), self.subject_id) if self.include_subject_ids else None
        val_subject_ids = np.full(len(y_val), self.subject_id) if self.include_subject_ids else None
        test_subject_ids = np.full(len(y_test), self.subject_id) if self.include_subject_ids else None

        # Create datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, train_subject_ids)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val, val_subject_ids)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, test_subject_ids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # keeps workers alive between epochs
                        #   prefetch_factor=4                 # each worker preloads 4 future batches                          
                        )


class BCICIV2aLOSO(BCICIV2a):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int, include_subject_ids=False):
        super().__init__(preprocessing_dict, subject_id, include_subject_ids)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        # Get session keys from first subject's split
        first_subj_split = splitted_ds[str(train_subjects[0])].split("session")
        train_key, test_key = _get_session_keys(first_subj_split)
        
        train_datasets = [splitted_ds[str(subj_id)].split("session")[train_key]
                            for subj_id in train_subjects]
        val_datasets = [splitted_ds[str(subj_id)].split("session")[test_key]
                        for subj_id in train_subjects]
        test_dataset = splitted_ds[str(self.subject_id)].split("session")[test_key]

        # load the data with subject IDs
        X_list, y_list, subj_list = [], [], []
        for subj_id, train_dataset in zip(train_subjects, train_datasets):
            X_part, y_part = _extract_data_from_dataset(train_dataset)
            X_list.append(X_part)
            y_list.append(y_part)
            if self.include_subject_ids:
                subj_list.append(np.full(len(y_part), subj_id))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        train_subject_ids = np.concatenate(subj_list, axis=0) if self.include_subject_ids else None
        
        X_val_list, y_val_list, val_subj_list = [], [], []
        for subj_id, val_dataset in zip(train_subjects, val_datasets):
            X_part, y_part = _extract_data_from_dataset(val_dataset)
            X_val_list.append(X_part)
            y_val_list.append(y_part)
            if self.include_subject_ids:
                val_subj_list.append(np.full(len(y_part), subj_id))
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        val_subject_ids = np.concatenate(val_subj_list, axis=0) if self.include_subject_ids else None
        
        X_test, y_test = _extract_data_from_dataset(test_dataset)
        test_subject_ids = np.full(len(y_test), self.subject_id) if self.include_subject_ids else None

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, train_subject_ids)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val, val_subject_ids)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, test_subject_ids)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # keeps workers alive between epochs
                        #   prefetch_factor=4                 # each worker preloads 4 future batches                          
                        )


class BCICIV2aAll(BaseDataModule):
    """
    Loads ALL subjects at once for joint training.

    Args:
        val_split: If a float in (0, 1), a stratified fraction of each subject's
                   training-session data is held out as a separate validation set.
                   When None (default), val_dataloader falls back to test_dataloader.
    """
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4
    num_subjects = 9

    def __init__(self, preprocessing_dict: dict, subject_id="all",
                 include_subject_ids=True, val_split: Optional[float] = None):
        # subject_id is ignored
        super().__init__(preprocessing_dict, subject_id=1)  # Dummy subject_id
        self.include_subject_ids = include_subject_ids
        self.val_split = val_split

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by subject first
        splitted_ds = self.dataset.split("subject")

        # Get session keys from first subject's split
        first_subj_split = splitted_ds[str(self.all_subject_ids[0])].split("session")
        train_key, test_key = _get_session_keys(first_subj_split)

        X_train_list, y_train_list, train_subj_list = [], [], []
        X_val_list,   y_val_list,   val_subj_list   = [], [], []
        X_test_list,  y_test_list,  test_subj_list  = [], [], []

        seed = self.preprocessing_dict.get("seed", 42)

        for subj_id in self.all_subject_ids:
            subj_data = splitted_ds[str(subj_id)].split("session")

            # Training session — optionally split into train + val
            X_part, y_part = _extract_data_from_dataset(subj_data[train_key])
            if self.val_split is not None:
                X_tr, X_v, y_tr, y_v = train_test_split(
                    X_part, y_part,
                    test_size=self.val_split,
                    random_state=seed,
                    stratify=y_part,
                )
                X_val_list.append(X_v)
                y_val_list.append(y_v)
                val_subj_list.append(np.full(len(y_v), subj_id))
            else:
                X_tr, y_tr = X_part, y_part
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            train_subj_list.append(np.full(len(y_tr), subj_id))

            # Test session
            X_part, y_part = _extract_data_from_dataset(subj_data[test_key])
            X_test_list.append(X_part)
            y_test_list.append(y_part)
            test_subj_list.append(np.full(len(y_part), subj_id))

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        train_subject_ids = np.concatenate(train_subj_list, axis=0)

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        test_subject_ids = np.concatenate(test_subj_list, axis=0)

        if self.val_split is not None:
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            val_subject_ids = np.concatenate(val_subj_list, axis=0)

            if self.preprocessing_dict["z_scale"]:
                X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

            self.val_dataset = BaseDataModule._make_tensor_dataset(
                X_val, y_val, val_subject_ids if self.include_subject_ids else None)
        else:
            if self.preprocessing_dict["z_scale"]:
                X_train, X_test = BaseDataModule._z_scale(X_train, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(
            X_train, y_train, train_subject_ids if self.include_subject_ids else None)
        self.test_dataset = BaseDataModule._make_tensor_dataset(
            X_test, y_test, test_subject_ids if self.include_subject_ids else None)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.preprocessing_dict["batch_size"],
                              num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                              pin_memory=True)
        return self.test_dataloader()
