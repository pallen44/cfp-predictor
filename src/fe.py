# src/fe.py
"""
Feature engineering: compute per-team, per-week summary statistics from raw games
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_DIR = Path(__file__).resolve().parents[1] / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def build_team_week_features(games_df: pd.DataFrame, season: int, rolling_window: int = 3) -> pd.DataFrame:
    """
    Compute team-week features (points, margins, win rates, rolling averages)
    Adapted for CFBD v2 schema with camelCase columns.
    """
    # Rename to consistent snake_case for easier handling
    games_df = games_df.rename(
        columns={
            "homeTeam": "home",
            "awayTeam": "away",
            "homePoints": "home_pts",
            "awayPoints": "away_pts",
            "neutralSite": "neutral",
            "week": "week",
            "seasonType": "season_type",
        }
    )

    # Filter only completed games with valid scores
    games_df = games_df[games_df["completed"] == True]
    games_df = games_df.dropna(subset=["home_pts", "away_pts"])

    # Flatten home/away perspectives
    home_rows = games_df[["home", "away", "home_pts", "away_pts", "week"]].copy()
    home_rows["team"] = home_rows["home"]
    home_rows["opp"] = home_rows["away"]
    home_rows["points_for"] = home_rows["home_pts"]
    home_rows["points_against"] = home_rows["away_pts"]
    home_rows["is_home"] = 1
    home_rows["win"] = (home_rows["home_pts"] > home_rows["away_pts"]).astype(int)

    away_rows = games_df[["away", "home", "away_pts", "home_pts", "week"]].copy()
    away_rows["team"] = away_rows["away"]
    away_rows["opp"] = away_rows["home"]
    away_rows["points_for"] = away_rows["away_pts"]
    away_rows["points_against"] = away_rows["home_pts"]
    away_rows["is_home"] = 0
    away_rows["win"] = (away_rows["away_pts"] > away_rows["home_pts"]).astype(int)

    # Combine home + away views
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games[["team", "opp", "week", "points_for", "points_against", "is_home", "win"]]
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]

    # Aggregate by team/week
    team_week = (
        team_games.groupby(["team", "week"])
        .agg(
            games_played=("opp", "count"),
            total_points_for=("points_for", "sum"),
            total_points_against=("points_against", "sum"),
            total_wins=("win", "sum"),
            avg_point_diff=("point_diff", "mean"),
            home_games=("is_home", "sum"),
        )
        .reset_index()
    )

    # Cumulative + rolling features
    team_week = team_week.sort_values(["team", "week"])
    team_week["cum_games"] = team_week.groupby("team")["games_played"].cumsum()
    team_week["cum_wins"] = team_week.groupby("team")["total_wins"].cumsum()
    team_week["cum_points_for"] = team_week.groupby("team")["total_points_for"].cumsum()
    team_week["cum_points_against"] = team_week.groupby("team")["total_points_against"].cumsum()

    team_week["win_rate"] = team_week["cum_wins"] / team_week["cum_games"]

    for col in ["avg_point_diff", "total_points_for", "total_points_against"]:
        team_week[f"{col}_roll{rolling_window}"] = (
            team_week.groupby("team")[col]
            .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        )

    team_week["season"] = season
    return team_week


def build_and_save_features(games_path: str, season: int, rolling_window: int = 3):
    df = pd.read_csv(games_path)
    features = build_team_week_features(df, season, rolling_window)
    out_path = FEATURE_DIR / f"team_week_features_{season}.parquet"
    features.to_parquet(out_path, index=False)
    print(f"✅ Saved features to {out_path}")
    return features
