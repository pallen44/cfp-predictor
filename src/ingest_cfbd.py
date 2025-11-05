# src/ingest_cfbd.py
"""
CFBD API ingestion utilities
Usage:
    from src.ingest_cfbd import CFBDClient
    cfbd = CFBDClient()
    games = cfbd.get_games(year=2024, season_type='regular')
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key
load_dotenv()
CFBD_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class CFBDClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or CFBD_KEY
        if not self.api_key:
            raise ValueError("CFBD_API_KEY not found in environment or .env file.")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _get(self, endpoint: str, params: dict = None) -> list:
        """Generic GET request handler"""
        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Request failed: {resp.status_code} - {resp.text}")
        return resp.json()

    def get_games(self, year: int, season_type: str = "regular") -> pd.DataFrame:
        """Fetch all games for a given year and season type."""
        data = self._get("games", {"year": year, "seasonType": season_type})
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_csv(DATA_DIR / f"games_{year}_{season_type}.csv", index=False)
        return df

    def get_team_stats(self, year: int, category: str = "team") -> pd.DataFrame:
        """Fetch basic team stats."""
        data = self._get("stats/season", {"year": year, "category": category})
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_csv(DATA_DIR / f"team_stats_{year}.csv", index=False)
        return df

    def get_teams(self) -> pd.DataFrame:
        data = self._get("teams/fbs")
        return pd.DataFrame(data)

    def get_upcoming_games(self, year: int, week: int) -> pd.DataFrame:
        """Get schedule for upcoming week."""
        data = self._get("games", {"year": year, "week": week, "seasonType": "regular"})
        return pd.DataFrame(data)

    def download_historical_games(self, start_year=2022, end_year=2024):
        """Download multiple seasons of games & cache locally."""
        all_games = []
        for y in tqdm(range(start_year, end_year + 1), desc="Downloading seasons"):
            df = self.get_games(y)
            all_games.append(df)
        all_df = pd.concat(all_games, ignore_index=True)
        all_df.to_csv(DATA_DIR / "games_all.csv", index=False)
        return all_df
