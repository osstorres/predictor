"""
ML model training, evaluation, and walk-forward backtesting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier


def prepare_model_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns
                    if c.startswith(("home_", "away_", "diff_",
                                     "norm_prob_", "odds_spread",
                                     "elo_", "h2h_",
                                     "rest_", "is_midweek"))]
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df["Result"].copy()
    print(f"Features: {X.shape[1]} | Matches: {X.shape[0]}")
    print(f"Class balance: {y.value_counts().sort_index().to_dict()} (0=Away, 1=Draw, 2=Home)")
    return X, y, feature_cols


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  eval_metric="mlogloss", verbosity=0),
    }

    results = {}
    for name, model in models.items():
        fold_accs, fold_lls = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr = scaler.fit_transform(X.iloc[train_idx])
            X_te = scaler.transform(X.iloc[test_idx])
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            proba = model.predict_proba(X_te)
            fold_accs.append(accuracy_score(y_te, preds))
            fold_lls.append(log_loss(y_te, proba))

        results[name] = {
            "accuracy_mean": np.mean(fold_accs),
            "accuracy_std": np.std(fold_accs),
            "log_loss_mean": np.mean(fold_lls),
        }
        print(f"  {name:25s}  Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}  "
              f"LogLoss: {np.mean(fold_lls):.4f}")

    return results, models


def build_ensemble(X: pd.DataFrame, y: pd.Series):
    scaler = StandardScaler()
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   random_state=42, eval_metric="mlogloss", verbosity=0)),
        ],
        voting="soft",
        weights=[1, 1, 2],
    )

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ensemble.fit(X_train_s, y_train)
    preds = ensemble.predict(X_test_s)
    proba = ensemble.predict_proba(X_test_s)

    print(f"\n  ENSEMBLE (Soft Voting)")
    print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"  Log Loss: {log_loss(y_test, proba):.4f}")
    print(classification_report(y_test, preds, target_names=["Away Win", "Draw", "Home Win"]))

    return ensemble, scaler


class WalkForwardBacktest:
    def __init__(self, initial_train_size: int = 500, step_size: int = 38):
        self.initial_train_size = initial_train_size
        self.step_size = step_size

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict:
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               random_state=42, eval_metric="mlogloss", verbosity=0)
        scaler = StandardScaler()
        all_preds, all_proba, all_true = [], [], []

        for start in range(self.initial_train_size, len(X) - self.step_size, self.step_size):
            end = start + self.step_size
            X_tr = scaler.fit_transform(X.iloc[:start])
            X_te = scaler.transform(X.iloc[start:end])
            y_tr, y_te = y.iloc[:start], y.iloc[start:end]
            model.fit(X_tr, y_tr)
            all_preds.extend(model.predict(X_te))
            all_proba.extend(model.predict_proba(X_te))
            all_true.extend(y_te.values)

        all_preds = np.array(all_preds)
        all_proba = np.array(all_proba)
        all_true = np.array(all_true)

        acc = accuracy_score(all_true, all_preds)
        ll = log_loss(all_true, all_proba)
        print(f"\nWalk-Forward Backtest | {len(all_preds)} predictions")
        print(f"  Accuracy: {acc:.4f} | Log Loss: {ll:.4f}")
        print(classification_report(all_true, all_preds, target_names=["Away", "Draw", "Home"]))

        return {"predictions": all_preds, "probabilities": all_proba, "actuals": all_true, "accuracy": acc}
