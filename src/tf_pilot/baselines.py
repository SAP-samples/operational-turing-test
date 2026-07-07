"""
Classifiers for the Operational Turing Test.

OracleClassifier   — executes rules.py; 100% accuracy (upper bound by construction).
XGBValuesOnly      — XGBoost on aggregate statistics of row values; no schema.
XGBRelationOnly    — XGBoost on values plus relation-only joins/group stats;
                     no executable rule residuals or legality predicates.
XGBSchemaAware     — same + cheap operational features from rules.py semantics.
TabICLValuesOnly   — TabICL v2 in-context learner on values-only features; free
                     HuggingFace download, no auth required, no training-size cap.
TabPFNValuesOnly   — TabPFN v2 in-context learner on values-only features; requires
                     TABPFN_TOKEN (https://ux.priorlabs.ai).
XGBOracleFeature   — XGBoost on values + oracle output as a feature; pipeline sanity
                     check (must reach 1.000 — proves labels and pipeline are consistent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# TabICL must be imported before XGBoost on macOS: XGBoost initialises OpenMP
# first, which conflicts with PyTorch's threading when loaded afterwards.
# The try/except keeps tabicl optional (not a hard dependency).
try:
    from tabicl import TabICLClassifier as _TabICLClassifier
    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False

from xgboost import XGBClassifier

from .rules import oracle
from .features import state_to_values_vector, state_to_relation_vector, state_to_schema_vector


class OracleClassifier:
    """
    Identifiability upper bound.  Runs the full rule set and returns 1 (legal)
    or 0 (illegal).  No training required; 100% accuracy by construction.
    """

    def fit(self, states: list[dict], labels: list[int]) -> "OracleClassifier":
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return np.array([1 if oracle(S)["legal"] else 0 for S in states], dtype=int)

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        preds = self.predict(states)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


class XGBValuesOnly:
    """
    XGBoost on aggregate statistics of row values only.
    No schema, no FK awareness, no rule knowledge.
    Expected accuracy: near 0.50 (empirical corollary of the identifiability claim).
    """

    def __init__(self, seed: int = 0) -> None:
        self._model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        self._fitted = False

    def _featurize(self, states: list[dict]) -> np.ndarray:
        return np.vstack([state_to_values_vector(S) for S in states])

    def fit(self, states: list[dict], labels: list[int]) -> "XGBValuesOnly":
        X = self._featurize(states)
        self._model.fit(X, np.array(labels))
        self._fitted = True
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))


class XGBRelationOnly:
    """
    XGBoost on value statistics plus relation-only structural features.

    This baseline has access to FK topology, joins, group-size distributions,
    and local relational neighborhoods, but not executable rule predicates such
    as derivation residuals or illegal-transition counts.
    """

    def __init__(self, seed: int = 0) -> None:
        self._model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        self._fitted = False

    def _featurize(self, states: list[dict]) -> np.ndarray:
        return np.vstack([state_to_relation_vector(S) for S in states])

    def fit(self, states: list[dict], labels: list[int]) -> "XGBRelationOnly":
        X = self._featurize(states)
        self._model.fit(X, np.array(labels))
        self._fitted = True
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))


class XGBSchemaAware:
    """
    XGBoost on values statistics plus cheap operational features.
    Any non-zero accuracy above chance is evidence that operational grounding
    helps — even a strawman version of it.
    """

    def __init__(self, seed: int = 0) -> None:
        self._model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        self._fitted = False

    def _featurize(self, states: list[dict]) -> np.ndarray:
        return np.vstack([state_to_schema_vector(S) for S in states])

    def fit(self, states: list[dict], labels: list[int]) -> "XGBSchemaAware":
        X = self._featurize(states)
        self._model.fit(X, np.array(labels))
        self._fitted = True
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))


class TabICLValuesOnly:
    """
    TabICL v2 in-context learner on values-only features.

    Free download from HuggingFace (jingang/TabICL), no auth required.
    Handles the full 2000-row training set natively (~6 s/seed on CPU).
    Architecture-independent test of the identifiability claim alongside XGBoost.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed   = seed
        self._model  = None

    def _featurize(self, states: list[dict]) -> np.ndarray:
        return np.vstack([state_to_values_vector(S) for S in states]).astype(np.float32)

    def fit(self, states: list[dict], labels: list[int]) -> "TabICLValuesOnly":
        if not _TABICL_AVAILABLE:
            raise RuntimeError("tabicl is not installed. Run: pip install tabicl")
        X = self._featurize(states)
        y = np.array(labels)
        self._model = _TabICLClassifier(random_state=self._seed, verbose=False, n_jobs=1)
        self._model.fit(X, y)
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))


class TabPFNValuesOnly:
    """
    TabPFN v1 (Hollmann et al. 2022) in-context learner on values-only features.

    Uses tabpfn<2: fully open-source, weights auto-download from HuggingFace
    (~35 MB, cached in ~/.cache/tabpfn/), no API key required.

    Hard limit: 1000 training samples and 100 features (v1 design constraint).
    We subsample to 1000 balanced examples when the dataset is larger.
    """

    MAX_TRAIN = 1000  # TabPFN v1 hard limit

    def __init__(self, seed: int = 0, device: str = "cpu") -> None:
        self._seed   = seed
        self._device = device
        self._model  = None
        self._fitted = False

    def _featurize(self, states: list[dict]) -> np.ndarray:
        return np.vstack([state_to_values_vector(S) for S in states])

    def fit(self, states: list[dict], labels: list[int]) -> "TabPFNValuesOnly":
        try:
            # tabpfn 0.1.x references Optional via torch internals removed in PyTorch >=2.x
            import torch.nn.modules.transformer as _t, typing
            if not hasattr(_t, "Optional"):
                _t.Optional = typing.Optional
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise RuntimeError("tabpfn is not installed. Run: pip install 'tabpfn==0.1.11'")

        X = self._featurize(states)
        y = np.array(labels)

        # Subsample to v1 limit, preserving class balance.
        if len(X) > self.MAX_TRAIN:
            rng  = np.random.default_rng(self._seed)
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            half = self.MAX_TRAIN // 2
            idx  = np.concatenate([
                rng.choice(idx0, min(half, len(idx0)), replace=False),
                rng.choice(idx1, min(half, len(idx1)), replace=False),
            ])
            X, y = X[idx], y[idx]

        # TabPFN v1 API: device and seed (not random_state).
        # Weights download automatically on first call (~35 MB, HuggingFace).
        self._model = TabPFNClassifier(
            device=self._device,
            N_ensemble_configurations=32,
            seed=self._seed,
        )
        self._model.fit(X, y, overwrite_warning=True)
        self._fitted = True
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))


class XGBOracleFeature:
    """
    Pipeline sanity check.

    XGBoost on values-only features plus one additional feature: the binary
    oracle output (1=legal, 0=illegal).  Must achieve accuracy = 1.000 because
    the oracle feature IS the label by construction.  A shortfall here indicates
    a bug in label generation or the oracle, not a modelling failure.
    """

    def __init__(self, seed: int = 0) -> None:
        self._model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.3,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )

    def _featurize(self, states: list[dict]) -> np.ndarray:
        vals      = np.vstack([state_to_values_vector(S) for S in states])
        oracle_f  = np.array([[1 if oracle(S)["legal"] else 0] for S in states], dtype=float)
        return np.hstack([vals, oracle_f])

    def fit(self, states: list[dict], labels: list[int]) -> "XGBOracleFeature":
        self._model.fit(self._featurize(states), np.array(labels))
        return self

    def predict(self, states: list[dict]) -> np.ndarray:
        return self._model.predict(self._featurize(states))

    def predict_proba(self, states: list[dict]) -> np.ndarray:
        return self._model.predict_proba(self._featurize(states))
