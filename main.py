"""
Football Match Prediction System
=================================
Combines: Historical stats + ELO + xG proxy + Bookmaker odds + Claude AI

Usage:
  python main.py               # full pipeline
  python main.py --quick       # skip backtest, use fewer seasons
  python main.py --predict     # predict + Claude report for upcoming matches
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── check ANTHROPIC_API_KEY ─────────────────────────────────────────────────
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key.")
    sys.exit(1)

from data_loader import FootballDataLoader, DataCleaner
from features import (
    FeatureEngineer, FootballELO,
    compute_xg_proxy, compute_fatigue_features,
    compute_h2h_features, add_odds_features,
)
from model import prepare_model_data, train_and_evaluate, build_ensemble, WalkForwardBacktest
from claude_analyst import (
    claude_analyze_matchup, generate_prediction_report, analyze_matchday,
)
from visualize import (
    plot_model_comparison, plot_confusion_matrix_heatmap,
    plot_feature_importance, plot_calibration_curve_chart,
)


# ── Config ────────────────────────────────────────────────────────────────────

SEASONS_FULL  = ["2425", "2324", "2223", "2122", "2021"]
SEASONS_QUICK = ["2425", "2324"]
LEAGUES       = ["E0", "SP1", "D1"]   # EPL, La Liga, Bundesliga


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(quick: bool = False, predict_demo: bool = False):

    seasons = SEASONS_QUICK if quick else SEASONS_FULL
    print(f"\n{'='*60}")
    print(f"  Football Prediction Pipeline  (quick={quick})")
    print(f"{'='*60}\n")

    # 1. Load & clean data
    print("── Step 1: Loading data ─────────────────────────────────────")
    loader = FootballDataLoader(seasons=seasons, leagues=LEAGUES)
    raw = loader.load_all()
    data = DataCleaner.clean(raw)
    print(f"Clean data: {len(data)} matches\n")

    # 2. Feature engineering
    print("── Step 2: Feature engineering ─────────────────────────────")
    engineer = FeatureEngineer(window=5)
    data = engineer.build_match_features(data)
    print(f"After rolling features: {len(data)} matches")

    elo = FootballELO(k=32, home_advantage=65)
    data = elo.compute_elo_features(data)

    top5 = sorted(elo.ratings.items(), key=lambda x: -x[1])[:5]
    print("Top 5 teams by ELO:")
    for team, rating in top5:
        print(f"  {team:25s}  {rating:.0f}")

    data = compute_xg_proxy(data)
    data = compute_fatigue_features(data)

    print("Computing H2H features (may take a moment)...")
    data = compute_h2h_features(data, n_last=5)
    data = add_odds_features(data)
    print(f"Total features ready: {len(data)} matches\n")

    # 3. Train models
    print("── Step 3: Training models ──────────────────────────────────")
    X, y, feature_names = prepare_model_data(data)
    print()
    results, models = train_and_evaluate(X, y)

    print("\n── Step 4: Ensemble model ───────────────────────────────────")
    ensemble, scaler = build_ensemble(X, y)

    # 4. Visualizations
    print("\n── Step 5: Visualizations ───────────────────────────────────")
    plot_model_comparison(results)

    # Predictions on test set for confusion matrix + calibration
    split_idx = int(len(X) * 0.8)
    X_test_s = scaler.transform(X.iloc[split_idx:])
    y_test = y.iloc[split_idx:]
    preds = ensemble.predict(X_test_s)
    proba = ensemble.predict_proba(X_test_s)
    plot_confusion_matrix_heatmap(y_test, preds)
    plot_calibration_curve_chart(y_test.values, proba, class_idx=2)

    # Feature importance from XGBoost sub-estimator
    try:
        xgb_est = next(e for name, e in ensemble.estimators_ if "xgb" in str(type(e)).lower())
        plot_feature_importance(xgb_est, feature_names, top_n=15)
    except (StopIteration, AttributeError):
        pass

    # 5. Walk-forward backtest
    if not quick:
        print("\n── Step 6: Walk-forward backtest ────────────────────────────")
        backtester = WalkForwardBacktest(initial_train_size=500, step_size=38)
        backtester.run(X, y)

    # 6. Demo: Claude reports for last 5 EPL matches
    if predict_demo:
        print("\n── Step 7: Claude AI analysis (demo on last 5 matches) ──────")
        recent = data[data["League"] == "Premier League"].tail(5)
        _demo_claude_reports(recent, ensemble, scaler, X.columns.tolist())

    print("\n✓ Pipeline complete.")
    return ensemble, scaler, data, feature_names


def _demo_claude_reports(df: pd.DataFrame, ensemble, scaler, feature_cols: list):
    """Generate Claude AI reports for a sample of matches."""
    matches_for_batch = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        feat_row = row[feature_cols].fillna(0).values.reshape(1, -1)
        feat_scaled = scaler.transform(feat_row)
        proba = ensemble.predict_proba(feat_scaled)[0]  # [away, draw, home]

        model_proba = {"home_win": proba[2], "draw": proba[1], "away_win": proba[0]}
        stats = {
            "home_avg_GF": row.get("home_avg_GF", 0),
            "home_avg_GA": row.get("home_avg_GA", 0),
            "home_avg_SoT": row.get("home_avg_SoT", 0),
            "home_Form": row.get("home_Form", 0),
            "away_avg_GF": row.get("away_avg_GF", 0),
            "away_avg_GA": row.get("away_avg_GA", 0),
            "away_avg_SoT": row.get("away_avg_SoT", 0),
            "away_Form": row.get("away_Form", 0),
        }

        print(f"\n  {home} vs {away}")
        print(f"  ML probs → Home: {proba[2]:.1%} | Draw: {proba[1]:.1%} | Away: {proba[0]:.1%}")

        report = generate_prediction_report(
            home_team=home, away_team=away,
            model_proba=model_proba, stats=stats,
            league=row.get("League", "Premier League"),
        )
        print(f"\n  {report}\n")
        print("  " + "-" * 50)

        matches_for_batch.append({
            "home": home, "away": away,
            "prob_H": proba[2], "prob_D": proba[1], "prob_A": proba[0],
            "home_form": stats["home_Form"],
            "away_form": stats["away_Form"],
        })

    print("\n── Matchday summary (Claude batch analysis) ─────────────────")
    summary = analyze_matchday(matches_for_batch)
    print(summary)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football prediction pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Use 2 seasons only (faster, for testing)")
    parser.add_argument("--predict", action="store_true",
                        help="Show Claude AI reports for recent matches")
    args = parser.parse_args()

    run_pipeline(quick=args.quick, predict_demo=args.predict)
