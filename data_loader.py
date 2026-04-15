"""
Data loading and cleaning for football match prediction.
Source: football-data.co.uk (free, no API key required)
"""

import pandas as pd
import numpy as np


class FootballDataLoader:
    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    LEAGUES = {
        "E0": "Premier League",
        "SP1": "La Liga",
        "D1": "Bundesliga",
        "I1": "Serie A",
        "F1": "Ligue 1",
    }

    COLUMNS_TO_KEEP = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS",
        "HST", "AST",
        "HF", "AF",
        "HC", "AC",
        "HY", "AY",
        "HR", "AR",
        "B365H", "B365D", "B365A",
    ]

    def __init__(self, seasons: list, leagues: list = None):
        self.seasons = seasons
        self.leagues = leagues or list(self.LEAGUES.keys())

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/{season}/{league}.csv"
        try:
            df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
            available_cols = [c for c in self.COLUMNS_TO_KEEP if c in df.columns]
            df = df[available_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df["League"] = self.LEAGUES.get(league, league)
            df["Season"] = season
            return df
        except Exception as e:
            print(f"  ✗ Error loading {league}/{season}: {e}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        frames = []
        for league in self.leagues:
            for season in self.seasons:
                df = self.load_season(league, season)
                if not df.empty:
                    frames.append(df)
                    print(f"  ✓ {self.LEAGUES.get(league)}, season {season}: {len(df)} matches")
        if not frames:
            raise ValueError("No data loaded. Check your seasons/leagues config.")
        result = pd.concat(frames, ignore_index=True)
        print(f"\nTotal loaded: {len(result)} matches")
        return result


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        numeric_cols = [
            "FTHG", "FTAG", "HTHG", "HTAG",
            "HS", "AS", "HST", "AST",
            "HF", "AF", "HC", "AC",
            "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        result_map = {"H": 2, "D": 1, "A": 0}
        df["Result"] = df["FTR"].map(result_map)
        df = df.dropna(subset=["Result"])
        df["Result"] = df["Result"].astype(int)
        return df
