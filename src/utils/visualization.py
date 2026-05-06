# IMPORTOWANIE BIBLIOTEK
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc
# CONFIG
import config


# USTALAMY WSPÓLNY STYL +-------------------------------------------------------

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

# ZAPISUJE JAKO PNG
def _save(fig: plt.Figure, filename: str) -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[✓] Plot saved → {path}")


# FUNKCJA STRATY +-------------------------------------------------------------

def plot_loss_curve(history: dict, save: bool = True) -> plt.Figure:
    """
    Plot training and validation loss over epochs.
    history must contain 'train_loss' and 'val_loss' keys.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train loss",  linewidth=2)
        ax.plot(epochs, history["val_loss"],   label="Val loss",    linewidth=2, linestyle="--")

        # Mark the best epoch (lowest val loss)
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val   = min(history["val_loss"])
        ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
        ax.scatter([best_epoch], [best_val], zorder=5, color="tab:orange", s=60)

        ax.set_title("Learning Curve — Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss")
        ax.legend()
        fig.tight_layout()

    if save:
        _save(fig, "loss_curve.png")
    return fig


# KRZYWA METRYK +-----------------------------------------------------------------

def plot_metrics_curve(history: dict, save: bool = True) -> plt.Figure:
    """
    Plot F1, Precision, Recall, and ROC-AUC over epochs on a single chart.
    """
    metric_keys = {
        "val_f1":        ("F1",        "tab:blue"),
        "val_precision": ("Precision", "tab:green"),
        "val_recall":    ("Recall",    "tab:red"),
        "val_roc_auc":   ("ROC-AUC",   "tab:purple"),
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        epochs = range(1, len(history["val_f1"]) + 1)

        for key, (label, color) in metric_keys.items():
            if key in history:
                ax.plot(epochs, history[key], label=label, color=color, linewidth=2)

        ax.set_title("Validation Metrics over Epochs", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend()
        fig.tight_layout()

    if save:
        _save(fig, "metrics_curve.png")
    return fig


# CONFUSION MATRIX (TABLICA POMYŁEK) +----------------------------------------------

def plot_confusion_matrix(
    cm:         np.ndarray,
    save:       bool = True,
    title:      str  = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a labelled, annotated confusion matrix heatmap.
    cm: 2x2 numpy array from sklearn.metrics.confusion_matrix
    """
    labels = ["Noise (0)", "Pulsar (1)"]
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct   = count / total * 100
                color = "white" if count > thresh else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", color=color, fontsize=12)

        ax.set_xticks([0, 1]);  ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]);  ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label",      fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Summary stats below the plot
        fig.text(0.5, -0.04,
                 f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
                 ha="center", fontsize=10, color="gray")
        fig.tight_layout()

    if save:
        _save(fig, "confusion_matrix.png")
    return fig


# KRZYWA ROC +--------------------------------------------------------------------------

def plot_roc_curve(
    y_true:  np.ndarray,
    y_probs: np.ndarray,
    save:    bool = True,
) -> plt.Figure:
    """
    Plot ROC curve with AUC annotation and the random-classifier baseline.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Find the point closest to the top-left corner (optimal threshold)
    optimal_idx   = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve  (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
                   color="tab:red", zorder=5, s=80,
                   label=f"Optimal threshold ≈ {optimal_thresh:.2f}")

        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
        fig.tight_layout()

    if save:
        _save(fig, "roc_curve.png")
    return fig


# STWÓRZ WSZYSTKIE WYKRESY +--------------------------------------------------------

def plot_all(history: dict, metrics: dict) -> None:
    """
    Generate all four plots in one call.
    Pass history from Trainer.fit() and metrics from evaluate_model().
    """
    plot_loss_curve(history)
    plot_metrics_curve(history)
    plot_confusion_matrix(metrics["confusion"])
    plot_roc_curve(metrics["y_true"], metrics["y_probs"])
    print("\n[✓] All plots saved to", config.PLOTS_DIR)