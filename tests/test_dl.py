"""Tests for src/models/dl_base.py, lstm.py, cnn.py, transformer.py."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.cnn import CNNModel
from src.models.dl_base import (
    SOHDataset,
    create_sequences,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_loop,
)
from src.models.lstm import LSTMModel
from src.models.transformer import PositionalEncoding, TransformerModel


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_feature_df():
    """Synthetic feature matrix for DL tests."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(50):
        records.append(
            {
                "cell_id": "cell_A",
                "cycle_number": i,
                "soh": 1.0 - 0.003 * i + rng.randn() * 0.01,
                "feat_1": rng.randn(),
                "feat_2": rng.randn(),
                "feat_3": rng.randn(),
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def windowed_data(synthetic_feature_df):
    """Create windowed sequences from the synthetic feature matrix."""
    feature_cols = ["feat_1", "feat_2", "feat_3"]
    X, y = create_sequences(synthetic_feature_df, feature_cols, window_size=5)
    return X, y


class TestSOHDataset:
    """Tests for the windowed sequence dataset."""

    def test_shapes(self, synthetic_feature_df):
        feature_cols = ["feat_1", "feat_2", "feat_3"]
        dataset = SOHDataset(synthetic_feature_df, feature_cols, window_size=5)
        assert len(dataset) == 46
        x, y = dataset[0]
        assert x.shape == (5, 3)
        assert y.ndim == 0

    def test_no_leakage(self, synthetic_feature_df):
        """Window should never include the target cycle's future data."""
        feature_cols = ["feat_1", "feat_2", "feat_3"]
        dataset = SOHDataset(synthetic_feature_df, feature_cols, window_size=5)
        for i in range(min(10, len(dataset))):
            x, y = dataset[i]
            last_cycle_feat = x[-1, 0].item()
            y.item()
            cell_df = synthetic_feature_df.sort_values("cycle_number")
            assert abs(last_cycle_feat - cell_df.iloc[i + 4]["feat_1"]) < 1e-5


class TestCreateSequences:
    """Tests for sequence creation utility."""

    def test_shapes(self, windowed_data):
        X, y = windowed_data
        assert X.ndim == 3
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 5
        assert X.shape[2] == 3

    def test_dtype(self, windowed_data):
        X, y = windowed_data
        assert X.dtype == np.float32
        assert y.dtype == np.float32


class TestModels:
    """Tests for model forward pass shapes."""

    def test_lstm_forward(self, device):
        model = LSTMModel(input_dim=3, hidden_1=32, hidden_2=16, dense_dim=8, dropout=0.1)
        x = torch.randn(4, 10, 3)
        out = model(x)
        assert out.shape == (4, 1)

    def test_cnn_forward(self, device):
        model = CNNModel(
            input_dim=3, filters_list=[16, 32, 64], kernel_size=3, dense_dim=16, dropout=0.1
        )
        x = torch.randn(4, 10, 3)
        out = model(x)
        assert out.shape == (4, 1)

    def test_transformer_forward(self, device):
        model = TransformerModel(
            input_dim=3, d_model=32, n_heads=2, n_blocks=1, dropout=0.1
        )
        x = torch.randn(4, 10, 3)
        out = model(x)
        assert out.shape == (4, 1)

    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=32, max_len=50)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)


class TestTrainLoop:
    """Tests for the training loop."""

    def test_converges(self, device):
        set_seed(42)
        model = LSTMModel(input_dim=3, hidden_1=16, hidden_2=8, dense_dim=4, dropout=0.0)
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5, 3).astype(np.float32)
        y = (0.8 + 0.1 * X[:, -1, 0]).astype(np.float32)

        dataset = list(
            zip(
                [torch.tensor(x) for x in X],
                [torch.tensor(v) for v in y],
                strict=False,
            )
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        config = {
            "learning_rate": 0.01,
            "max_epochs": 20,
            "patience_early_stopping": 20,
            "patience_lr_reduce": 10,
            "lr_reduce_factor": 0.5,
        }
        history = train_loop(model, loader, loader, config, device, seed=42)
        assert len(history["train_loss"]) > 5
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_early_stopping(self, device):
        set_seed(42)
        model = LSTMModel(input_dim=2, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        X_train = np.random.RandomState(0).randn(30, 5, 2).astype(np.float32)
        y_train = np.random.RandomState(0).randn(30).astype(np.float32)
        X_val = np.random.RandomState(1).randn(10, 5, 2).astype(np.float32)
        y_val = np.random.RandomState(1).randn(10).astype(np.float32)
        train_ds = list(
            zip(
                [torch.tensor(x) for x in X_train], [torch.tensor(v) for v in y_train], strict=False
            )
        )
        val_ds = list(
            zip([torch.tensor(x) for x in X_val], [torch.tensor(v) for v in y_val], strict=False)
        )
        train_loader = DataLoader(train_ds, batch_size=16)
        val_loader = DataLoader(val_ds, batch_size=10)
        config = {
            "learning_rate": 0.001,
            "max_epochs": 200,
            "patience_early_stopping": 3,
            "patience_lr_reduce": 2,
            "lr_reduce_factor": 0.5,
        }
        history = train_loop(model, train_loader, val_loader, config, device, seed=42)
        assert len(history["train_loss"]) < 200


class TestEvaluate:
    """Tests for evaluation utility."""

    def test_returns_arrays(self, device):
        model = LSTMModel(input_dim=3, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        X = np.random.randn(10, 5, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        dataset = list(
            zip([torch.tensor(x) for x in X], [torch.tensor(v) for v in y], strict=False)
        )
        loader = DataLoader(dataset, batch_size=5)
        y_true, y_pred = evaluate(model, loader, device)
        assert len(y_true) == 10
        assert len(y_pred) == 10


class TestSeedReproducibility:
    """Tests for deterministic seeding."""

    def test_same_seed_same_weights(self, device):
        set_seed(42)
        model1 = LSTMModel(input_dim=3, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        w1 = model1.lstm1.weight_ih_l0.clone()

        set_seed(42)
        model2 = LSTMModel(input_dim=3, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        w2 = model2.lstm1.weight_ih_l0.clone()

        assert torch.equal(w1, w2)


class TestCheckpoint:
    """Tests for save/load checkpoint."""

    def test_roundtrip(self, device, tmp_path):
        set_seed(42)
        model = LSTMModel(input_dim=3, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        path = tmp_path / "model.pt"
        save_checkpoint(model, str(path))
        assert path.exists()

        model2 = LSTMModel(input_dim=3, hidden_1=8, hidden_2=4, dense_dim=2, dropout=0.0)
        load_checkpoint(model2, str(path), device)
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
            assert torch.equal(p1, p2)
