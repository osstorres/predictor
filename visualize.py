"""
Visualization utilities: model comparison, confusion matrix,
feature importance, calibration curve, probability distributions.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.size"] = 11
sns.set_style("whitegrid")


def plot_model_comparison(results: dict, save_path: str = "model_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(results.keys())
    accuracies = [results[n]["accuracy_mean"] for n in names]
    acc_stds = [results[n]["accuracy_std"] for n in names]
    log_losses = [results[n]["log_loss_mean"] for n in names]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][:len(names)]

    bars = axes[0].barh(names, accuracies, xerr=acc_stds, color=colors, edgecolor="white")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Model Accuracy (TimeSeriesSplit CV)")
    axes[0].set_xlim(0.3, 0.65)
    for bar, val in zip(bars, accuracies):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontweight="bold")

    bars = axes[1].barh(names, log_losses, color=colors, edgecolor="white")
    axes[1].set_xlabel("Log Loss (lower = better)")
    axes[1].set_title("Model Log Loss")
    for bar, val in zip(bars, log_losses):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix_heatmap(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Away Win", "Draw", "Home Win"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.75, f"({cm_pct[i,j]:.0%})",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Ensemble Model")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15,
                             save_path: str = "feature_importance.png"):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), importances[indices], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_calibration_curve_chart(y_true, y_proba, class_idx: int = 2,
                                  class_name: str = "Home Win",
                                  save_path: str = "calibration_curve.png"):
    prob_true, prob_pred = calibration_curve(
        (y_true == class_idx).astype(int),
        y_proba[:, class_idx],
        n_bins=10, strategy="uniform",
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="#e74c3c",
            label=f"Model ({class_name})", linewidth=2, markersize=8)
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_triple_layer_radar(match_name: str, bookmaker: dict,
                             polymarket: dict, ml_model: dict,
                             save_path: str = "triple_radar.png"):
    categories = ["Home Win", "Draw", "Away Win"]
    keys = ["home", "draw", "away"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    for label, probs, color in [
        ("Bookmaker", bookmaker, "#3498db"),
        ("Polymarket", polymarket, "#e74c3c"),
        ("ML Model", ml_model, "#2ecc71"),
    ]:
        values = [probs.get(k, 0) for k in keys] + [probs.get(keys[0], 0)]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Triple Layer: {match_name}", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
