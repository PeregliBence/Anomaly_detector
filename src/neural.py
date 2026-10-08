"""Neural architectures: a feed-forward network and an LSTM.

Both implement the same Model interface as everything else, so they run
through the identical walk-forward, purge and scoring. Every learned model is
given the same columns: the LSTM reads the `seq_` lag columns as a sequence,
the tabular models see them as flat features. The comparison then isolates
the architecture's inductive bias rather than rewarding whichever model was
handed more information.

Everything that learns from data — clipping bounds, scaling, early-stopping
point — is fitted inside `fit`, on the training fold only.

Early stopping needs its own purge. The validation slice is the most recent
part of the training window; if it began right after the training rows, their
overlapping labels would reach into it and the stopping point would be chosen
on leaked information. A gap of `horizon` dates separates the two.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.models import Model

logger = logging.getLogger(__name__)

SEQ_PATTERN = re.compile(r"^seq_(?P<channel>[a-z]+)_(?P<lag>\d+)$")


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Preprocessor:
    """Clip to the training fold's 1st/99th percentiles, then standardise."""

    def fit(self, X: np.ndarray) -> "_Preprocessor":
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        clipped = np.clip(X, self.lo_, self.hi_)
        self.mean_ = clipped.mean(axis=0)
        self.std_ = clipped.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.clip(X, self.lo_, self.hi_) - self.mean_) / self.std_).astype(np.float32)


def _time_split(X: pd.DataFrame, val_frac: float, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for train and validation, separated by `gap` dates."""
    dates = X.index.get_level_values("date")
    unique = dates.unique().sort_values()
    n_val = max(int(len(unique) * val_frac), 20)
    val_start = unique[-n_val]
    train_end = unique[max(len(unique) - n_val - gap, 1)]
    return np.asarray(dates < train_end), np.asarray(dates >= val_start)


class _TorchClassifier(Model):
    """Shared training loop: Adam, BCE, early stopping, seed ensemble."""

    needs_horizon = True

    def __init__(self, horizon: int = 20, seeds: int = 3, epochs: int = 20,
                 batch_size: int = 4096, lr: float = 1e-3, patience: int = 3,
                 val_frac: float = 0.1, weight_decay: float = 1e-5):
        self.horizon = horizon
        self.seeds = seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self.weight_decay = weight_decay

    # Subclasses: split columns, build tensors, build the network.
    def _prepare(self, X: pd.DataFrame, fit: bool):
        raise NotImplementedError

    def _network(self):
        raise NotImplementedError

    def _forward(self, net, batch):
        raise NotImplementedError

    def _batches(self, tensors, idx):
        return [t[idx] for t in tensors]

    # Pickling stores weights on the CPU and rebuilds the networks on load, so
    # a model trained on a GPU loads on a machine without one, and the LSTM's
    # locally defined network class never has to be pickled itself.
    def __getstate__(self):
        state = self.__dict__.copy()
        nets = state.pop("nets_", None)
        if nets is not None:
            state["net_states_"] = [
                {k: v.detach().cpu() for k, v in net.state_dict().items()} for net in nets
            ]
        return state

    def __setstate__(self, state):
        net_states = state.pop("net_states_", None)
        self.__dict__.update(state)
        if net_states is not None:
            device = _device()
            self.nets_ = []
            for weights in net_states:
                net = self._network()
                net.load_state_dict(weights)
                self.nets_.append(net.to(device).eval())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchClassifier":
        torch = _torch()
        device = _device()

        train_mask, val_mask = _time_split(X, self.val_frac, self.horizon)
        tensors = self._prepare(X, fit=True)
        target = torch.tensor(y.values.astype(np.float32), device=device)
        train_idx = torch.tensor(np.flatnonzero(train_mask), device=device)
        val_idx = torch.tensor(np.flatnonzero(val_mask), device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self.nets_ = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = self._network().to(device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            best_loss, best_state, stale = np.inf, None, 0
            for _ in range(self.epochs):
                net.train()
                order = train_idx[torch.randperm(len(train_idx), device=device)]
                for start in range(0, len(order), self.batch_size):
                    idx = order[start:start + self.batch_size]
                    opt.zero_grad()
                    logits = self._forward(net, self._batches(tensors, idx))
                    loss_fn(logits, target[idx]).backward()
                    opt.step()

                val_loss = self._loss(net, tensors, target, val_idx, loss_fn)
                if val_loss < best_loss - 1e-5:
                    best_loss, stale = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.patience:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
        return self

    def _loss(self, net, tensors, target, idx, loss_fn) -> float:
        torch = _torch()
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(idx), self.batch_size * 4):
                b = idx[start:start + self.batch_size * 4]
                total += loss_fn(self._forward(net, self._batches(tensors, b)), target[b]).item() * len(b)
                n += len(b)
        return total / max(n, 1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        tensors = self._prepare(X, fit=False)
        n = len(X)
        out = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for net in self.nets_:
                parts = []
                for start in range(0, n, self.batch_size * 4):
                    idx = torch.arange(start, min(start + self.batch_size * 4, n), device=_device())
                    parts.append(torch.sigmoid(self._forward(net, self._batches(tensors, idx))).cpu().numpy())
                out += np.concatenate(parts)
        return out / len(self.nets_)


class MLP(_TorchClassifier):
    """Feed-forward network in the style of Gu, Kelly and Xiu's NN3.

    A pyramid of hidden layers with batch normalisation and dropout. All
    columns, including the sequence lags, enter as one flat vector.
    """

    name = "mlp"

    def __init__(self, hidden=(64, 32, 16), dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.dropout = dropout

    def _prepare(self, X, fit):
        torch = _torch()
        values = X.values.astype(np.float32)
        if fit:
            self.pre_ = _Preprocessor().fit(values)
            self.n_inputs_ = values.shape[1]
        return [torch.tensor(self.pre_.transform(values), device=_device())]

    def _network(self):
        nn = _torch().nn
        layers, width = [], self.n_inputs_
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _forward(self, net, batch):
        return net(batch[0]).squeeze(-1)


class LSTM(_TorchClassifier):
    """Recurrent branch over the daily sequences plus a tabular branch.

    The `seq_<channel>_<lag>` columns are reshaped to (batch, time, channel)
    with the oldest day first, read by an LSTM, and its final state is joined
    with an encoding of today's tabular features.
    """

    name = "lstm"

    def __init__(self, hidden: int = 32, **kwargs):
        kwargs.setdefault("seeds", 2)
        kwargs.setdefault("epochs", 15)
        super().__init__(**kwargs)
        self.hidden = hidden

    def _layout(self, columns):
        seq = {}
        for col in columns:
            m = SEQ_PATTERN.match(col)
            if m:
                seq.setdefault(m["channel"], {})[int(m["lag"])] = col
        if not seq:
            raise KeyError("LSTM needs seq_<channel>_<lag> columns; build the panel with seq_len > 0")
        channels = sorted(seq)
        lags = sorted(next(iter(seq.values())), reverse=True)  # oldest first
        ordered = [seq[c][lag] for lag in lags for c in channels]
        tabular = [c for c in columns if not SEQ_PATTERN.match(c)]
        return ordered, tabular, len(lags), len(channels)

    def _prepare(self, X, fit):
        torch = _torch()
        if fit:
            self.seq_cols_, self.tab_cols_, self.steps_, self.channels_ = self._layout(list(X.columns))
            self.seq_pre_ = _Preprocessor().fit(X[self.seq_cols_].values.astype(np.float32))
            self.tab_pre_ = _Preprocessor().fit(X[self.tab_cols_].values.astype(np.float32))

        seq = self.seq_pre_.transform(X[self.seq_cols_].values.astype(np.float32))
        seq = seq.reshape(len(X), self.steps_, self.channels_)
        tab = self.tab_pre_.transform(X[self.tab_cols_].values.astype(np.float32))
        device = _device()
        return [torch.tensor(seq, device=device), torch.tensor(tab, device=device)]

    def _network(self):
        nn = _torch().nn
        hidden, n_tab, channels = self.hidden, len(self.tab_cols_), self.channels_

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(channels, hidden, batch_first=True)
                self.tab = nn.Sequential(nn.Linear(n_tab, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
                self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                          nn.Dropout(0.1), nn.Linear(hidden, 1))

            def forward(self, seq, tab):
                _, (h, _) = self.rnn(seq)
                return self.head(_torch().cat([h[-1], self.tab(tab)], dim=1)).squeeze(-1)

        return Net()

    def _forward(self, net, batch):
        return net(batch[0], batch[1])
