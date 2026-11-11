import os

import numpy as np
import torch

from module.data_manager.manager import DataManager


class IMDB(DataManager):
    def __init__(
        self,
        checkpoint="distilbert-base-uncased",
        max_length=256,
        encode_batch_size=32,
        dataset_dir=None,
        model_dir=None,
        preprocessed_path=None,
    ):
        super().__init__(name="IMDB")
        self.task = "TextClassification"
        self.checkpoint = checkpoint
        self.max_length = int(max_length)
        self.encode_batch_size = int(encode_batch_size)
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.preprocessed_path = preprocessed_path

    def _resolve_project_root(self):
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    def _default_preprocessed_path(self):
        file_name = f"imdb_{self.checkpoint.replace('/', '_')}_maxlen{self.max_length}.pt"
        return os.path.join(self._resolve_project_root(), "data", "raw", "imdb_preprocessed", file_name)

    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        preprocessed_path = self.preprocessed_path or self._default_preprocessed_path()

        if not os.path.isfile(preprocessed_path):
            raise FileNotFoundError(
                f"Preprocessed IMDB embeddings not found: {preprocessed_path}. "
                f"Please run script/preprocess_imdb_embeddings.py first."
            )

        payload = torch.load(preprocessed_path, map_location="cpu")
        if "X" not in payload or "y" not in payload:
            raise ValueError(
                f"Invalid preprocessed IMDB file: {preprocessed_path}. "
                f"Expected keys 'X' and 'y'."
            )

        X = payload["X"].float()
        y = payload["y"].long()

        if nrows is not None:
            limit = min(int(nrows), len(y))
            X = X[:limit]
            y = y[:limit]

        n = len(y)
        rng = np.random.RandomState(shuffle_seed)
        indices = rng.permutation(n)

        n_test = int(n * float(test_ratio))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

        self.X = X
        self.y = y

        if cuda:
            self._data_to_cuda()
