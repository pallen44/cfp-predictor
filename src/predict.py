# src/predict.py
import pandas as pd
import numpy as np
from src.elo import EloModel
from src.fe import make_game_features
from src.train_base import fit_glm, fit_gbm
from src.stack import stack_models

def train_and_predict(past_games, team_week_features, future_games):
    # Fit Elo on past results
    elo = EloModel()
    for wk in sorted(past_games['week'].unique()):
        wk_games = past_games[past_games['week']==wk]
        for r in wk_games.itertuples(index=False):
            elo.update_game(r.home, r.away, r.home_pts, r.away_pts, r.neutral)
        elo.decay_all()

    # Build features for past games (for learning y = home_win)
    X_past, meta_past = make_game_features(
        past_games[['game_id','week','home','away','neutral']], team_week_features, elo
    )
    y = (past_games['home_pts'] > past_games['away_pts']).astype(int).values

    # Base models & stacking
    base_fns = [fit_glm, fit_gbm]
    models, stacker = stack_models(X_past.values, y, meta_past, base_fns)

    # Predict future weeks
    X_future, meta_future = make_game_features(future_games, team_week_features, elo)
    # base predictions
    P_base = []
    for m in models:
        P_base.append(m.predict_proba(np.nan_to_num(X_future))[:,1])
    P_base = np.column_stack(P_base)
    p_final = stacker.predict_proba(P_base)[:,1]

    return meta_future.assign(p_home_win=p_final)
