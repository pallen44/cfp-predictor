# src/stack.py
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def time_split(df, time_col, n_folds=4):
    # rolling-origin splits by week
    weeks = sorted(df[time_col].unique())
    folds = []
    for i in range(n_folds):
        cut = int(len(weeks)*(i+1)/(n_folds+1))
        train_weeks = weeks[:cut]
        valid_weeks = [weeks[cut]]
        folds.append((train_weeks, valid_weeks))
    return folds

def stack_models(X, y, meta, base_models_fit_fn):
    oof = []
    weights_train = []
    folds = time_split(meta, 'week', n_folds=4)
    for train_weeks, valid_weeks in folds:
        tr = meta['week'].isin(train_weeks)
        va = meta['week'].isin(valid_weeks)
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        models = [fn(X_tr, y_tr) for fn in base_models_fit_fn]
        P = np.column_stack([m.predict_proba(X_va)[:,1] if hasattr(m,'predict_proba') else m(X_va) for m in models])
        oof.append(pd.DataFrame(P, columns=[f'm{i}' for i in range(len(models))]).assign(y=y_va.values, idx=np.where(va)[0]))

    oof_df = pd.concat(oof, axis=0).sort_values('idx')
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(oof_df[[c for c in oof_df.columns if c.startswith('m')]], oof_df['y'])

    # final refit on full data
    final_models = [fn(X, y) for fn in base_models_fit_fn]

    return final_models, stacker
