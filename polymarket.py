"""
Polymarket Gamma API client — no API key required.
Fetches football market probabilities from the blockchain prediction market.
"""

import requests
import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/118.0.0.0 Safari/537.36"
    )
}

FOOTBALL_KEYWORDS = [
    "soccer", "premier league", "la liga", "bundesliga",
    "serie a", "ligue 1", "champions league", "uefa",
    "manchester", "liverpool", "arsenal", "chelsea",
    "barcelona", "real madrid", "bayern", "psg",
    "epl", "football match",
]


@dataclass
class PolymarketOdds:
    home_win: float
    away_win: float
    liquidity: float
    volume_24h: float
    market_slug: str
    last_updated: str
    draw: float = None


class PolymarketClient:
    def search_football_markets(self, limit: int = 100) -> list:
        all_markets = []
        offset = 0
        while offset < limit:
            try:
                resp = requests.get(
                    f"{GAMMA_API}/markets",
                    params={"active": "true", "closed": "false", "limit": 50, "offset": offset},
                    headers=HEADERS, timeout=15,
                )
                resp.raise_for_status()
                markets = resp.json()
                if not markets:
                    break
                for market in markets:
                    text = (market.get("question", "") + " " + market.get("description", "")).lower()
                    if any(kw in text for kw in FOOTBALL_KEYWORDS):
                        all_markets.append(market)
                offset += 50
                time.sleep(0.5)
            except requests.RequestException as e:
                print(f"  Warning: {e}")
                break
        print(f"  Found {len(all_markets)} football markets on Polymarket")
        return all_markets

    def extract_match_odds(self, market: dict):
        try:
            outcomes = market.get("outcomes", [])
            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            if len(prices) < 2:
                return None
            prices = [float(p) for p in prices]
            outcomes_lower = [o.lower() for o in outcomes]

            if len(prices) == 2:
                return PolymarketOdds(
                    home_win=prices[0], draw=None, away_win=prices[1],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    last_updated=market.get("updatedAt", ""),
                )

            if len(prices) >= 3:
                home_idx = next((i for i, o in enumerate(outcomes_lower) if "home" in o or "win" in o), 0)
                draw_idx = next((i for i, o in enumerate(outcomes_lower) if "draw" in o or "tie" in o), 1)
                away_idx = next((i for i, o in enumerate(outcomes_lower) if "away" in o or "lose" in o), 2)
                return PolymarketOdds(
                    home_win=prices[home_idx], draw=prices[draw_idx], away_win=prices[away_idx],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    last_updated=market.get("updatedAt", ""),
                )
        except (ValueError, IndexError, KeyError):
            return None


class TripleLayerFeatures:
    @staticmethod
    def compute_divergence_features(bookmaker_probs: dict, polymarket_probs: dict, ml_probs: dict = None) -> dict:
        features = {}
        epsilon = 1e-6

        for prefix, probs in [("bk", bookmaker_probs), ("poly", polymarket_probs)]:
            features[f"{prefix}_prob_H"] = probs.get("home", 0)
            features[f"{prefix}_prob_D"] = probs.get("draw", 0)
            features[f"{prefix}_prob_A"] = probs.get("away", 0)

        kl_div = 0
        for key in ["home", "draw", "away"]:
            p = max(bookmaker_probs.get(key, epsilon), epsilon)
            q = max(polymarket_probs.get(key, epsilon), epsilon)
            kl_div += p * np.log(p / q)
        features["kl_div_bk_poly"] = kl_div

        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            bk = bookmaker_probs.get(key, 0)
            poly = polymarket_probs.get(key, 0)
            features[f"divergence_{label}"] = bk - poly
            features[f"abs_divergence_{label}"] = abs(bk - poly)

        features["max_divergence"] = max(
            features["abs_divergence_H"],
            features["abs_divergence_D"],
            features["abs_divergence_A"],
        )

        bk_favorite = max(bookmaker_probs, key=bookmaker_probs.get)
        poly_favorite = max(polymarket_probs, key=polymarket_probs.get)
        features["sources_agree"] = int(bk_favorite == poly_favorite)

        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            features[f"blended_prob_{label}"] = 0.5 * bookmaker_probs.get(key, 0) + 0.5 * polymarket_probs.get(key, 0)

        if ml_probs:
            for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
                ml = ml_probs.get(key, 0)
                bk = bookmaker_probs.get(key, 0)
                poly = polymarket_probs.get(key, 0)
                features[f"ml_prob_{label}"] = ml
                features[f"ml_vs_bk_{label}"] = ml - bk
                features[f"ml_vs_poly_{label}"] = ml - poly
                features[f"triple_blend_{label}"] = 0.40 * ml + 0.35 * poly + 0.25 * bk
            ml_favorite = max(ml_probs, key=ml_probs.get)
            features["all_three_agree"] = int(bk_favorite == poly_favorite == ml_favorite)

        return features
