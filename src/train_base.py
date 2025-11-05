# src/train_base.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

def fit_glm(X_train, y_train):
    glm = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    glm.fit(np.nan_to_num(X_train), y_train)
    return glm

def fit_gbm(X_train, y_train):
    gbm = LGBMClassifier(
        n_estimators=800, learning_rate=0.02, max_depth=-1, subsample=0.9, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_samples=50
    )
    gbm.fit(np.nan_to_num(X_train), y_train)
    return gbm

def probs(model, X):
    return model.predict_proba(np.nan_to_num(X))[:,1]
