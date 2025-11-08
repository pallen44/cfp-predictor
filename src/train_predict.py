# src/train_predict.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


from src.datasets import prepare_training_set, prepare_prediction_set, save_predictions

@dataclass
class Models:
    lr: LogisticRegression
    gbm: LGBMClassifier

def train_models(X_train, y_train) -> Models:
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)

    gbm = LGBMClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=-1,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        min_child_samples=40
    )
    gbm.fit(X_train, y_train)

    return Models(lr=lr, gbm=gbm)

def predict_ensemble(models: Models, X) -> np.ndarray:
    p_lr = models.lr.predict_proba(X)[:,1]
    p_gbm = models.gbm.predict_proba(X)[:,1]
    # Simple average ensemble (strong & stable out of the box)
    return 0.5 * (p_lr + p_gbm)

def evaluate(models: Models, X, y) -> Tuple[float,float,float]:
    p = predict_ensemble(models, X)
    return (
        log_loss(y, p),
        brier_score_loss(y, p),
        accuracy_score(y, (p>=0.5).astype(int))
    )

def train_eval_split_by_week(X, y, meta, val_weeks=2):
    """
    Use the last `val_weeks` of available weeks for validation.
    Includes guards for empty or malformed splits.
    """
    if "week" not in meta.columns:
        raise ValueError("meta DataFrame missing 'week' column for split logic.")

    # Flatten and ensure numeric weeks
    weeks_series = pd.Series(meta["week"]).squeeze()
    weeks_series = pd.to_numeric(weeks_series, errors="coerce")
    weeks_sorted = sorted(weeks_series.dropna().unique())

    if len(weeks_sorted) < val_weeks:
        raise ValueError(f"Not enough distinct weeks ({len(weeks_sorted)}) for val_weeks={val_weeks}")

    train_weeks = weeks_sorted[:-val_weeks]
    valid_weeks = weeks_sorted[-val_weeks:]

    tr_mask = meta["week"].isin(train_weeks).to_numpy()
    va_mask = meta["week"].isin(valid_weeks).to_numpy()

    if not tr_mask.any() or not va_mask.any():
        raise ValueError("No data available for either training or validation split. Check 'week' values.")

    return (X[tr_mask], y[tr_mask]), (X[va_mask], y[va_mask]), train_weeks, valid_weeks


def run_training_pipeline(
    games_csv: str, features_parquet: str, season: int, val_weeks: int = 2
):
    X, y, meta, feat_cols = prepare_training_set(games_csv, features_parquet, season)
    (X_tr, y_tr), (X_va, y_va), train_weeks, valid_weeks = train_eval_split_by_week(X, y, meta, val_weeks=val_weeks)

    models = train_models(X_tr, y_tr)
    ll, br, acc = evaluate(models, X_va, y_va)
    print(f"Validation (weeks {valid_weeks}) — LogLoss: {ll:.4f}  Brier: {br:.4f}  Acc: {acc:.3f}")
    return models

def run_prediction_pipeline(
    models: Models, games_csv: str, features_parquet: str, season: int, weeks, out_stub="predictions"
):
    Xp, meta, _ = prepare_prediction_set(games_csv, features_parquet, season, weeks=weeks)
    p = predict_ensemble(models, Xp)
    out_name = f"{out_stub}_weeks_{'-'.join(map(str, sorted(weeks)))}.csv"
    return save_predictions(meta, p, out_name)
