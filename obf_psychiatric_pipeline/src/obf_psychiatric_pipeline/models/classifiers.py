"""
Classifier factory functions — returns sklearn-compatible estimators.

Design contract:
    - One factory function per classifier.
    - Each returns a sklearn Pipeline (preprocessor + estimator).
    - All estimators are reproducible via seed.
    - No fitting happens here — factories only build unfitted objects.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_dummy(seed: int = 42) -> DummyClassifier:
    """Floor baseline: stratified random guessing."""
    return DummyClassifier(strategy="stratified", random_state=seed)


def make_logreg(seed: int = 42) -> Pipeline:
    """Interpretable linear baseline with standard scaling and balanced weights."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )),
    ])


def make_xgb(seed: int = 42, n_classes: int = 3) -> XGBClassifier:
    """Strong non-linear baseline — gradient boosted trees."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=seed,
        verbosity=0,
    )