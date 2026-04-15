"""
Standalone: predict a single upcoming match using Claude + Polymarket + manual stats.

Usage:
  python predict_match.py --home Arsenal --away Chelsea --league "Premier League"
"""

import argparse
import json
from dotenv import load_dotenv
from claude_analyst import claude_analyze_matchup, generate_prediction_report, claude_analyze_divergence
from polymarket import PolymarketClient
from visualize import plot_triple_layer_radar

load_dotenv()


def predict_single_match(home: str, away: str, league: str,
                          home_form: dict = None, away_form: dict = None):
    # Defaults if no form data provided
    home_form = home_form or {"avg_GF": 1.5, "avg_GA": 1.0, "avg_SoT": 4.5, "Form": 1.8}
    away_form = away_form or {"avg_GF": 1.2, "avg_GA": 1.3, "avg_SoT": 3.8, "Form": 1.5}

    print(f"\n{'='*60}")
    print(f"  {home} vs {away} | {league}")
    print(f"{'='*60}")

    # 1. Claude contextual analysis
    print("\n── Claude contextual analysis ───────────────────────────────")
    analysis = claude_analyze_matchup(home, away, home_form, away_form, league)
    if analysis:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))

    # Build rough probabilities from Claude scores
    home_win_conf = analysis.get("home_win_confidence", 0.45)
    draw_like = analysis.get("draw_likelihood", 0.25)
    away_win = max(0.05, 1 - home_win_conf - draw_like)

    ml_probs = {"home": home_win_conf, "draw": draw_like, "away": away_win}

    # 2. Bookmaker placeholder (would come from live odds API)
    bookmaker = {"home": 0.48, "draw": 0.27, "away": 0.25}

    # 3. Polymarket — search for this match
    print("\n── Searching Polymarket ─────────────────────────────────────")
    poly_client = PolymarketClient()
    markets = poly_client.search_football_markets(limit=100)
    query = f"{home.lower()} {away.lower()}"
    match_markets = [
        m for m in markets
        if home.lower() in m.get("question", "").lower()
        or away.lower() in m.get("question", "").lower()
    ]
    print(f"  Found {len(match_markets)} relevant Polymarket market(s) for this match")

    poly_probs = {"home": 0.44, "draw": 0.26, "away": 0.30}  # fallback
    poly_liquidity = 0
    poly_volume = 0
    for m in match_markets[:3]:
        odds = poly_client.extract_match_odds(m)
        if odds:
            poly_probs["home"] = odds.home_win
            poly_probs["away"] = odds.away_win
            poly_probs["draw"] = odds.draw or max(0, 1 - odds.home_win - odds.away_win)
            poly_liquidity = odds.liquidity
            poly_volume = odds.volume_24h
            print(f"  Market: {m.get('question', '')}")
            print(f"  Home: {odds.home_win:.1%} | Draw: {poly_probs['draw']:.1%} | Away: {odds.away_win:.1%}")
            print(f"  Liquidity: ${poly_liquidity:,.0f} | 24h Vol: ${poly_volume:,.0f}")
            break

    # 4. Divergence analysis via Claude
    print("\n── Divergence analysis ──────────────────────────────────────")
    divergence_report = claude_analyze_divergence(
        match=f"{home} vs {away}",
        bookmaker=bookmaker,
        polymarket=poly_probs,
        ml_model=ml_probs,
        poly_liquidity=poly_liquidity,
        poly_volume_24h=poly_volume,
    )
    print(divergence_report)

    # 5. Full prediction report
    print("\n── Full prediction report ───────────────────────────────────")
    report = generate_prediction_report(
        home_team=home, away_team=away,
        model_proba={"home_win": ml_probs["home"], "draw": ml_probs["draw"], "away_win": ml_probs["away"]},
        stats={
            "home_avg_GF": home_form["avg_GF"], "home_avg_GA": home_form["avg_GA"],
            "home_avg_SoT": home_form["avg_SoT"], "home_Form": home_form["Form"],
            "away_avg_GF": away_form["avg_GF"], "away_avg_GA": away_form["avg_GA"],
            "away_avg_SoT": away_form["avg_SoT"], "away_Form": away_form["Form"],
        },
        league=league,
    )
    print(report)

    # 6. Radar chart
    plot_triple_layer_radar(
        match_name=f"{home} vs {away}",
        bookmaker=bookmaker,
        polymarket=poly_probs,
        ml_model=ml_probs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default="Arsenal", help="Home team name")
    parser.add_argument("--away", default="Chelsea", help="Away team name")
    parser.add_argument("--league", default="Premier League", help="League name")
    args = parser.parse_args()

    predict_single_match(
        home=args.home,
        away=args.away,
        league=args.league,
    )
