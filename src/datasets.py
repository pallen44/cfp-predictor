# src/datasets.py
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS_BASE = [
    "win_rate",
    "avg_point_diff_roll3",
    "total_points_for_roll3",
    "total_points_against_roll3",
]
# You can add more later; keep them numeric.

def _asof_join(games, feats, side_prefix, team_col, week_col="week"):
    """
    Reliable previous-week join that avoids merge_asof's strict sorting rules.
    Finds each team's most recent available features before the current game week.
    """

    feats = feats.copy()
    games = games.copy()

    # Ensure numeric week values
    feats["week"] = pd.to_numeric(feats["week"], errors="coerce")
    games[week_col] = pd.to_numeric(games[week_col], errors="coerce")

    # Rename feature columns
    feats = feats.rename(
        columns={c: f"{side_prefix}_{c}" for c in FEATURE_COLS_BASE + ["team", "week"]}
    )

    # Merge on team to get all candidate feature rows
    merged = games.merge(
        feats,
        left_on=team_col,
        right_on=f"{side_prefix}_team",
        how="left",
    )

    # Keep only rows where feature week < game week
    merged = merged[merged[f"{side_prefix}_week"] < merged[week_col]]

    # For each (team, game_week), keep the row with the max available feature week
    idx = (
        merged.groupby([team_col, week_col])[f"{side_prefix}_week"]
        .idxmax()
        .dropna()
    )

    merged_final = merged.loc[idx].copy()

    # Drop duplicate helper columns
    keep_cols = games.columns.tolist() + [
        f"{side_prefix}_{c}" for c in FEATURE_COLS_BASE
    ]
    merged_final = merged_final[keep_cols].reset_index(drop=True)

    return merged_final


def _make_design_matrix(games_df, feats_df):
    """
    Returns X (numpy array), y (for training), meta (home, away, week, neutral)
    games_df is already filtered to the rows we want (train or predict).
    feats_df = team-week features for that season.
    """
    # Ensure required cols exist
    for c in ["home", "away", "week", "neutral"]:
        if c not in games_df.columns:
            raise KeyError(f"Missing column in games_df: {c}")

    # Left-asof join for home/away features as of week-1
    home_merged = _asof_join(games_df, feats_df, "home", "home")
    away_merged = _asof_join(games_df, feats_df, "away", "away")

    # Stitch columns back onto games
    meta = games_df.reset_index(drop=True).copy()
    meta = pd.concat([meta, home_merged, away_merged], axis=1)

    # Feature diffs (home - away) for the base numeric features
    for base in FEATURE_COLS_BASE:
        meta[f"diff_{base}"] = meta[f"home_{base}"] - meta[f"away_{base}"]

    # Add context features (neutral)
    # Convert to int safely — fill NaN with 0 (non-neutral)
    meta["neutral"] = meta["neutral"].fillna(0).astype(int)

    # Build X
    feat_cols = [f"diff_{b}" for b in FEATURE_COLS_BASE] + ["neutral"]
    X = meta[feat_cols].fillna(0.0).to_numpy(dtype=float)

    # Target (if available)
    y = None
    if "home_pts" in games_df.columns and "away_pts" in games_df.columns:
        y = (games_df["home_pts"].values > games_df["away_pts"].values).astype(int)

    return X, y, meta[["home", "away", "week", "neutral"]], feat_cols

def prepare_training_set(games_csv_path: str, features_parquet_path: str, season: int):
    games = pd.read_csv(games_csv_path)
    feats = pd.read_parquet(features_parquet_path)

    # Normalize names to our expected set
    games = games.rename(columns={
        "homeTeam":"home", "awayTeam":"away",
        "homePoints":"home_pts", "awayPoints":"away_pts",
        "neutralSite":"neutral", "seasonType":"season_type"
    })

    # Train on completed regular-season games for this season
    g = games[(games["season"]==season) & (games["season_type"]=="regular") & (games["completed"]==True)].copy()
    # Only the columns we need downstream
    g = g[["home","away","week","neutral","home_pts","away_pts"]].copy()

    # Asof-join + design
    X, y, meta, feat_cols = _make_design_matrix(g, feats)

    # --- ensure single clean numeric 'week' column ---
    if "week" in meta.columns:
      # if it's a DataFrame (duplicate col), keep the first and drop others
      if isinstance(meta["week"], pd.DataFrame):
        meta["week"] = meta["week"].iloc[:, 0]
      elif isinstance(meta["week"], pd.Series):
        meta["week"] = pd.to_numeric(meta["week"], errors="coerce")
    else:
      meta["week"] = pd.to_numeric(g["week"], errors="coerce")

    # Drop duplicate columns if any still exist
    meta = meta.loc[:, ~meta.columns.duplicated()].copy()
    # ---------------------------------------------------

    return X, y, meta, feat_cols



def prepare_prediction_set(games_csv_path: str, features_parquet_path: str, season: int, weeks=None):
    games = pd.read_csv(games_csv_path)
    feats = pd.read_parquet(features_parquet_path)

    games = games.rename(columns={
        "homeTeam":"home", "awayTeam":"away",
        "neutralSite":"neutral", "seasonType":"season_type"
    })

    g = games[(games["season"]==season) & (games["season_type"]=="regular") & (games["completed"]==False)].copy()
    if weeks:
        g = g[g["week"].isin(weeks)].copy()

    g = g[["home","away","week","neutral"]].copy()
    X, y_dummy, meta, feat_cols = _make_design_matrix(g, feats)
    return X, meta, feat_cols

def save_predictions(meta_df, probs, out_name: str):
    # Drop duplicate columns, preserving order
    out = meta_df.loc[:, ~meta_df.columns.duplicated()].copy()

    # Ensure canonical column order
    keep_cols = [c for c in ["home", "away", "week", "neutral"] if c in out.columns]
    out = out[keep_cols].copy()

    # Attach prediction column
    out["p_home_win"] = probs

    # Save to disk
    out_path = PROC_DIR / out_name
    out.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    return out

