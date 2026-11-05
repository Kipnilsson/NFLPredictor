"""Model definitions and hyperparameter grids for the specialist classifiers."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class MajorityBaseline:
    """Predicts the training-set dropback rate for every play."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])

    def get_params(self, deep=True):
        return {}


def make_logreg(C: float) -> Pipeline:
    # Encoders/scalers live inside the pipeline, so they are fit on training data only.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), config.CATEGORICAL),
        ("num", StandardScaler(), config.NUMERIC),
    ])
    # L2 penalty: sklearn>=1.8 deprecates `penalty=`; the default (l1_ratio=0) is pure L2.
    clf = LogisticRegression(C=C, l1_ratio=0.0, max_iter=2000, random_state=config.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_gbt(learning_rate: float, max_leaf_nodes: int, min_samples_leaf: int):
    # Categorical columns are integer-coded (down 1-4, qtr 1-5); no scaling for trees.
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_iter=1000,
        early_stopping=True,          # internal holdout carved from TRAINING data only
        validation_fraction=0.1,
        n_iter_no_change=30,
        categorical_features=config.CATEGORICAL,
        random_state=config.SEED,
    )


LOGREG_GRID = [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0]]
GBT_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl}
    for lr, leaves, msl in itertools.product([0.03, 0.1], [15, 31, 63], [20, 100])
]

FACTORIES = {"logreg": make_logreg, "gbt": make_gbt}
GRIDS = {"logreg": LOGREG_GRID, "gbt": GBT_GRID}
