"""Sequence model adapters (§10.3, Phase 13).

Five architectures — a single LSTM, a stacked LSTM, a bidirectional LSTM, a GRU
and a small TCN — behind the same `AlphaModel` interface every other model uses,
so they land on the same leaderboard against the same baselines and the same
economic objective. Torch is imported lazily inside this module and lives in the
optional `dl` extra; `import alphacore` on a machine without it still works, and
`tests/unit/test_import_isolation.py` fails the build if that ever stops being
true.

Three things are load-bearing here, and all three are places a sequence model
leaks if you look away:

**Windows are cut on the bar grid, not on row positions.** A fold's training
rows arrive purged, so the frame has a hole where the test block was. The window
builder in `alphacore.models.windows` refuses to lay a window across that hole,
which is why `set_bar_grid` must be called before `fit` — without the full
index there is no way to tell a purge gap from an adjacent bar. It raises rather
than guessing.

**The preprocessor is fitted inside `fit`.** Same winsoriser and scaler as every
other adapter (§10.1, I3), fitted on the fold's training rows and nothing else.
Standardising a sequence globally "because the network needs it" is the most
common way this layer leaks in published notebooks.

**Early stopping uses an inner tail split, never the test fold** (§10.2). The
last `val_fraction` of the training windows is held out for the stopping rule,
the learning-rate schedule and the probability calibrator. A random inner split
would put windows that overlap the validation rows into training, which is the
same overlap problem the outer folds exist to solve.

Determinism (§4, I8): CPU by default, one thread, seeded torch and seeded batch
order. A GPU can be requested explicitly and the run is then reproducible only
on that machine, which is why it is never chosen automatically (ADR 0011).
"""

from __future__ import annotations

import copy
import pickle
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from alphacore.errors import ModelError
from alphacore.models.base import (
    CALIBRATION_METHODS,
    apply_calibrator,
    as_label_series,
    fit_calibrator,
    make_preprocessor,
)
from alphacore.models.protocol import Task
from alphacore.models.windows import build_windows, grid_positions
from alphacore.runtime.determinism import deterministic_rng
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = [
    "ARCHITECTURES",
    "SEQUENCE_DEFAULTS",
    "SequenceModel",
    "make_bilstm",
    "make_gru",
    "make_lstm",
    "make_stacked_lstm",
    "make_tcn",
]

ARCHITECTURES: Final[tuple[str, ...]] = ("lstm", "gru", "tcn")
"""Network families. `lstm_stacked` and `bilstm` are the LSTM with different
depth and direction settings, not separate architectures — one code path is one
place for the window logic to be wrong."""

SEQUENCE_DEFAULTS: Final[dict[str, Any]] = {
    "architecture": "lstm",
    "seq_len": 32,
    "hidden_size": 32,
    "num_layers": 1,
    "bidirectional": False,
    "dropout": 0.1,
    "channels": 16,
    "kernel_size": 3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 40,
    "patience": 6,
    "batch_size": 64,
    "val_fraction": 0.25,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "device": "cpu",
    "checkpoint_dir": None,
}
"""Small on purpose (§19). A 200k-parameter network on 2,000 daily bars will
reach a training accuracy that means nothing at all."""

_MIN_TRAIN_WINDOWS: Final[int] = 32
_MIN_VAL_WINDOWS: Final[int] = 16
_MIN_DELTA: Final[float] = 1e-5
_N_CLASSES: Final[int] = 2


def _require_torch() -> Any:  # noqa: ANN401
    """Import torch, or explain how to install it.

    Returns:
        The torch module.

    Raises:
        ModelError: If torch is not installed.
    """
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        msg = (
            "torch is not installed. Sequence models ship in the optional 'dl' extra: "
            "run `uv sync --all-packages --extra dl`."
        )
        raise ModelError(msg) from exc


def _resolve_device(torch: Any, requested: str) -> Any:  # noqa: ANN401
    """Pick the device to train on.

    Args:
        torch: The torch module.
        requested: `cpu`, `cuda`, `mps`, or `auto`.

    Returns:
        A `torch.device`.

    Raises:
        ModelError: If the name is unknown or the accelerator is unavailable.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            chosen = "cuda"
        elif torch.backends.mps.is_available():
            chosen = "mps"
        else:
            return torch.device("cpu")
    elif requested in {"cuda", "mps"}:
        chosen = requested
    else:
        msg = f"Unknown device {requested!r}. Known: cpu, cuda, mps, auto."
        raise ModelError(msg)

    available = torch.cuda.is_available() if chosen == "cuda" else torch.backends.mps.is_available()
    if not available:
        msg = f"Device {chosen!r} was requested but is not available on this machine."
        raise ModelError(msg)
    logger.bind(phase="train").warning(
        "Training on {}. Floating-point reductions on an accelerator are not bit-reproducible "
        "across machines, so artifacts from this run satisfy I8 only on this one.",
        chosen,
    )
    return torch.device(chosen)


class SequenceModel:
    """A recurrent or convolutional network over fold-aware windows."""

    task: Task = "binary"

    def __init__(
        self,
        name: str,
        *,
        seed: int,
        calibrate: str = "none",
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the network without building it.

        The network itself is constructed in `fit`, once the feature count is
        known and the seed can be applied immediately before initialisation.

        Args:
            name: Model name, as it appears on the leaderboard.
            seed: Run seed.
            calibrate: One of `none`, `sigmoid`, `isotonic`.
            params: Overrides merged onto `SEQUENCE_DEFAULTS`.

        Raises:
            ModelError: If the architecture, calibration method, or a numeric
                setting is invalid.
        """
        settings = {**SEQUENCE_DEFAULTS, **(params or {})}
        unknown = sorted(set(settings) - set(SEQUENCE_DEFAULTS))
        if unknown:
            msg = f"Unknown sequence parameter(s) {unknown} for model {name!r}."
            raise ModelError(msg)
        if settings["architecture"] not in ARCHITECTURES:
            msg = (
                f"Unknown architecture {settings['architecture']!r} for model {name!r}. "
                f"Known: {list(ARCHITECTURES)}."
            )
            raise ModelError(msg)
        if calibrate not in CALIBRATION_METHODS:
            msg = (
                f"Unknown calibration method {calibrate!r} for model {name!r}. "
                f"Known: {list(CALIBRATION_METHODS)}."
            )
            raise ModelError(msg)
        if int(settings["seq_len"]) < 1:
            msg = f"seq_len must be at least 1 bar, got {settings['seq_len']}."
            raise ModelError(msg)
        if int(settings["max_epochs"]) < 1:
            msg = f"max_epochs must be at least 1, got {settings['max_epochs']}."
            raise ModelError(msg)

        self.name = name
        self.seed = seed
        self.calibrate = calibrate
        self.params = settings
        self.seq_len = int(settings["seq_len"])
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        self.n_train_windows = 0
        self._grid: pd.Index | None = None
        self._pipeline: Pipeline | None = None
        self._net: Any = None
        self._calibrator: Any = None
        self._columns: list[str] = []

    def __repr__(self) -> str:
        """Return a debug representation naming the architecture and window."""
        return (
            f"SequenceModel({self.name!r}, architecture={self.params['architecture']!r}, "
            f"seq_len={self.seq_len})"
        )

    def set_bar_grid(self, index: pd.Index) -> None:
        """Record the complete bar index the folds were cut from.

        Args:
            index: The full feature index, not the fold's rows. Windows are cut
                on positions in this index, which is the only way to tell a
                purged block from an adjacent bar.
        """
        self._grid = pd.Index(index)

    def _require_grid(self) -> pd.Index:
        """Return the bar grid.

        Returns:
            The grid.

        Raises:
            ModelError: If it was never set.
        """
        if self._grid is None:
            msg = (
                f"Sequence model {self.name!r} needs the full bar grid before it can fit: call "
                "set_bar_grid(features.index) first. Without it a window cannot tell a purged "
                "block from the next bar, and would train across the test fold (§10.3). "
                "alphacore.models.fit_predict_oof does this for you."
            )
            raise ModelError(msg)
        return self._grid

    def _require_fitted(self) -> Any:  # noqa: ANN401
        """Return the trained network.

        Returns:
            The torch module.

        Raises:
            ModelError: If the model has not been fitted.
        """
        if self._net is None:
            msg = f"Model {self.name!r} must be fitted before it can predict."
            raise ModelError(msg)
        return self._net

    def _build_net(self, torch: Any, n_features: int) -> Any:  # noqa: ANN401
        """Construct the network for a known feature count.

        Args:
            torch: The torch module.
            n_features: Features per bar.

        Returns:
            An unfitted `torch.nn.Module`.
        """
        from alphacore.models import _nets  # noqa: PLC0415

        torch.manual_seed(self.seed)
        architecture = str(self.params["architecture"])
        if architecture == "tcn":
            return _nets.TemporalConvNet(
                n_features,
                channels=int(self.params["channels"]),
                num_layers=int(self.params["num_layers"]) + 1,
                kernel_size=int(self.params["kernel_size"]),
                dropout=float(self.params["dropout"]),
            )
        return _nets.RecurrentClassifier(
            n_features,
            cell=architecture,
            hidden_size=int(self.params["hidden_size"]),
            num_layers=int(self.params["num_layers"]),
            bidirectional=bool(self.params["bidirectional"]),
            dropout=float(self.params["dropout"]),
        )

    def _windows_for(
        self, X: pd.DataFrame, *, pad: bool, fit_preprocessor: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Preprocess a frame and cut it into fold-aware windows.

        Args:
            X: The rows to window.
            pad: Whether every row must produce a window (§10.3).
            fit_preprocessor: Whether to fit the winsoriser and scaler. True
                only inside `fit`, on training rows only (I3).

        Returns:
            `(windows, rows)` as returned by `build_windows`.

        Raises:
            ModelError: If the rows are not on the bar grid.
        """
        grid = self._require_grid()
        if fit_preprocessor:
            self._pipeline = Pipeline(make_preprocessor(scale=True))
            matrix = np.asarray(self._pipeline.fit_transform(X), dtype=float)
        else:
            assert self._pipeline is not None
            matrix = np.asarray(self._pipeline.transform(X), dtype=float)
        positions = grid_positions(X.index, grid)
        return build_windows(matrix, positions, seq_len=self.seq_len, pad=pad)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: NDArray[np.float64] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,  # noqa: ARG002
    ) -> SequenceModel:
        """Train the network on one fold's rows.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Per-sample weights (§8.2), applied to the loss of the
                window each row ends.
            eval_set: Unused. The stopping split is taken from the tail of the
                training windows, because an externally supplied validation set
                would not carry the window purge.

        Returns:
            Self, fitted.

        Raises:
            ModelError: If the grid was never set, the inputs disagree in
                length, the fold is single-class, or too few windows survive.
        """
        torch = _require_torch()
        if len(X) != len(y):
            msg = f"Model {self.name!r} got {len(X)} rows of features for {len(y)} labels."
            raise ModelError(msg)
        labels = as_label_series(y, X.index)
        if len(np.unique(labels.to_numpy())) < _N_CLASSES:
            msg = f"Model {self.name!r} needs both classes in the training fold."
            raise ModelError(msg)
        self._columns = [str(c) for c in X.columns]

        windows, rows = self._windows_for(X, pad=False, fit_preprocessor=True)
        if windows.shape[0] < _MIN_TRAIN_WINDOWS:
            msg = (
                f"Model {self.name!r} has only {windows.shape[0]} complete windows of "
                f"{self.seq_len} bars in a fold of {len(X)} rows. Shorten seq_len or widen the "
                "training window; padding the shortfall would invent bars."
            )
            raise ModelError(msg)
        self.n_train_windows = int(windows.shape[0])

        targets = labels.to_numpy(dtype=float)[rows]
        weights = (
            np.ones(rows.size, dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)[rows]
        )
        self._train(torch, windows, targets, weights)
        return self

    def _train(
        self,
        torch: Any,  # noqa: ANN401
        windows: NDArray[np.float64],
        targets: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Run the training loop with early stopping and a plateau schedule.

        Args:
            torch: The torch module.
            windows: `(n, seq_len, n_features)` training windows.
            targets: Binary targets for each window.
            weights: Per-window loss weights.
        """
        torch.set_num_threads(1)
        device = _resolve_device(torch, str(self.params["device"]))
        net = self._build_net(torch, windows.shape[2]).to(device)

        n_val = int(len(windows) * float(self.params["val_fraction"]))
        has_validation = n_val >= _MIN_VAL_WINDOWS
        cut = len(windows) - n_val if has_validation else len(windows)

        def split(values: NDArray[np.float64]) -> tuple[Any, Any]:
            return (
                torch.tensor(values[:cut], dtype=torch.float32, device=device),
                torch.tensor(values[cut:], dtype=torch.float32, device=device),
            )

        train_x, val_x = split(windows)
        train_y, val_y = split(targets)
        train_w, val_w = split(weights)

        optimiser = torch.optim.Adam(
            net.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            factor=float(self.params["lr_factor"]),
            patience=int(self.params["lr_patience"]),
        )
        rng = deterministic_rng(self.seed, "sequence", self.name, "batch-order")

        best_loss = np.inf
        best_state = copy.deepcopy(net.state_dict())
        self.best_epoch = 0
        self.history = []
        stale = 0

        for epoch in range(1, int(self.params["max_epochs"]) + 1):
            train_loss = self._run_epoch(torch, net, optimiser, (train_x, train_y, train_w), rng)
            net.eval()
            with torch.no_grad():
                val_loss = (
                    float(_weighted_bce(torch, net(val_x), val_y, val_w).item())
                    if has_validation
                    else train_loss
                )
            scheduler.step(val_loss)
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": float(optimiser.param_groups[0]["lr"]),
                }
            )

            if val_loss < best_loss - _MIN_DELTA:
                best_loss, stale = val_loss, 0
                self.best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                self._checkpoint(torch, best_state)
                continue
            stale += 1
            if stale >= int(self.params["patience"]):
                logger.bind(phase="train").debug(
                    "{}: stopped at epoch {}, best was {}", self.name, epoch, self.best_epoch
                )
                break

        net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self._device = device
        # Nothing ever improved on the initial weights; those are what we have.
        self.best_epoch = self.best_epoch or 1
        if has_validation and self.calibrate != "none":
            with torch.no_grad():
                raw = _to_probability(torch, net(val_x))
            self._calibrator = fit_calibrator(
                self.calibrate, raw, pd.Series(targets[cut:].astype(int))
            )

    def _run_epoch(
        self,
        torch: Any,  # noqa: ANN401
        net: Any,  # noqa: ANN401
        optimiser: Any,  # noqa: ANN401
        batch_data: tuple[Any, Any, Any],
        rng: np.random.Generator,
    ) -> float:
        """Take one pass over the training windows in a seeded random order.

        Args:
            torch: The torch module.
            net: The network being trained.
            optimiser: Its optimiser.
            batch_data: `(windows, targets, weights)` as tensors.
            rng: The seeded generator deciding batch order, so two runs with the
                same seed see the same batches in the same order (I8).

        Returns:
            The weighted mean training loss over the epoch.
        """
        train_x, train_y, train_w = batch_data
        net.train()
        order = rng.permutation(len(train_x))
        batch_size = int(self.params["batch_size"])
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = torch.tensor(order[start : start + batch_size], device=train_x.device)
            optimiser.zero_grad()
            loss = _weighted_bce(torch, net(train_x[batch]), train_y[batch], train_w[batch])
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(batch)
        return total / max(len(order), 1)

    def _checkpoint(self, torch: Any, state: dict[str, Any]) -> None:  # noqa: ANN401
        """Write the best weights to disk when a checkpoint directory is set.

        Args:
            torch: The torch module.
            state: The state dict to persist.
        """
        directory = self.params["checkpoint_dir"]
        if directory is None:
            return
        from pathlib import Path  # noqa: PLC0415

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(state, target / f"{self.name}.pt")

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Score every row, padding the short history at the head of a run.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array whose rows sum to 1.

        Raises:
            ModelError: If the model has not been fitted, or a row is not on the
                bar grid.
        """
        torch = _require_torch()
        net = self._require_fitted()
        windows, rows = self._windows_for(X, pad=True, fit_preprocessor=False)
        if rows.size != len(X):  # pragma: no cover - defensive
            msg = f"Windowing {len(X)} rows produced {rows.size} predictions."
            raise ModelError(msg)

        net.eval()
        scores = np.empty(rows.size, dtype=float)
        batch_size = int(self.params["batch_size"])
        with torch.no_grad():
            for start in range(0, rows.size, batch_size):
                batch = torch.tensor(
                    windows[start : start + batch_size], dtype=torch.float32, device=self._device
                )
                scores[start : start + batch_size] = _to_probability(torch, net(batch))

        if self._calibrator is not None:
            scores = apply_calibrator(self._calibrator, scores)
        positive = np.clip(scores, 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Predict class labels at a 0.5 threshold.

        Args:
            X: Feature frame.

        Returns:
            An `(n,)` integer array.

        Raises:
            ModelError: If the model has not been fitted.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # noqa: PLR2004

    def feature_importance(self) -> pd.Series | None:
        """Report nothing, deliberately.

        A recurrent network has no per-feature importance worth the name. Input
        gradients or permutation importance would both be defensible numbers,
        but they answer a different question from the tree importances the
        leaderboard shows beside them, and putting them in the same column would
        invite exactly the comparison they do not support (§7.3).

        Returns:
            None.
        """
        self._require_fitted()
        return None

    def save(self, path: Path) -> None:
        """Persist the fitted model with pickle.

        The whole object is pickled — preprocessor, network weights and
        calibrator — for the reason §7.10 of the handoff gives: a bare
        `state_dict` restores a network whose inputs are unscaled and whose
        outputs are not calibrated probabilities.

        Args:
            path: Destination file.

        Raises:
            ModelError: If the model has not been fitted.
        """
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> SequenceModel:
        """Load a persisted sequence model.

        Args:
            path: Source file, written by `save`.

        Returns:
            The restored model.

        Raises:
            ModelError: If the file does not contain one.
        """
        obj: Any = pickle.loads(path.read_bytes())
        if not isinstance(obj, SequenceModel):
            msg = f"{path} does not contain an alphacore sequence model."
            raise ModelError(msg)
        return obj


def _weighted_bce(
    torch: Any,  # noqa: ANN401
    logits: Any,  # noqa: ANN401
    targets: Any,  # noqa: ANN401
    weights: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Binary cross-entropy weighted by §8.2 sample weights.

    Args:
        torch: The torch module.
        logits: Raw network outputs.
        targets: Binary targets.
        weights: Per-sample weights.

    Returns:
        The mean weighted loss, as a tensor.
    """
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp(min=1e-12)


def _to_probability(torch: Any, logits: Any) -> NDArray[np.float64]:  # noqa: ANN401
    """Squash logits to probabilities as a numpy array.

    Args:
        torch: The torch module.
        logits: Raw network outputs.

    Returns:
        Probabilities in `[0, 1]`.
    """
    return np.asarray(torch.sigmoid(logits).detach().cpu().numpy(), dtype=float)


def _make_sequence(
    name: str,
    *,
    seed: int,
    calibrate: str,
    params: Mapping[str, Any] | None,
    **preset: Any,  # noqa: ANN401
) -> SequenceModel:
    """Build a sequence model from a preset plus config overrides.

    Args:
        name: Model name.
        seed: Random seed.
        calibrate: Calibration method.
        params: Config overrides, which win over the preset.
        **preset: The variant's own settings.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    _require_torch()
    return SequenceModel(name, seed=seed, calibrate=calibrate, params={**preset, **(params or {})})


def make_lstm(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> SequenceModel:
    """Build a single-layer LSTM.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto `SEQUENCE_DEFAULTS`.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    return _make_sequence(
        "lstm", seed=seed, calibrate=calibrate, params=params, architecture="lstm"
    )


def make_stacked_lstm(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> SequenceModel:
    """Build a two-layer LSTM.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto `SEQUENCE_DEFAULTS`.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    return _make_sequence(
        "lstm_stacked",
        seed=seed,
        calibrate=calibrate,
        params=params,
        architecture="lstm",
        num_layers=2,
    )


def make_bilstm(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> SequenceModel:
    """Build a bidirectional LSTM over the trailing window.

    Reading the window backwards is legal because the window ends at the row
    being scored (§10.3). It is worth stating plainly: bidirectionality is a
    leak only when the window itself contains the future, and the window builder
    makes that impossible.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto `SEQUENCE_DEFAULTS`.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    return _make_sequence(
        "bilstm",
        seed=seed,
        calibrate=calibrate,
        params=params,
        architecture="lstm",
        bidirectional=True,
    )


def make_gru(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> SequenceModel:
    """Build a GRU.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto `SEQUENCE_DEFAULTS`.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    return _make_sequence("gru", seed=seed, calibrate=calibrate, params=params, architecture="gru")


def make_tcn(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> SequenceModel:
    """Build a small dilated causal TCN.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto `SEQUENCE_DEFAULTS`.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If torch is missing or a setting is invalid.
    """
    return _make_sequence("tcn", seed=seed, calibrate=calibrate, params=params, architecture="tcn")
