# IMPORTOWANIE BIBLIOTEK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
# CONFIG
import config


# EWALUACJA MODELU +---------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference over the full loader and return all metrics + raw arrays.

    Returns:
        loss        : not computed here (no criterion) — set to None
        f1          : F1 score (binary)
        precision   : precision
        recall      : recall  ← most important metric for this problem
        roc_auc     : ROC-AUC
        confusion   : 2x2 confusion matrix (np.ndarray)
        report      : full sklearn classification_report string
        y_true      : ground truth labels (np.ndarray)
        y_probs     : predicted probabilities for class 1 (np.ndarray)
        y_preds     : binary predictions at config.THRESHOLD (np.ndarray)
    """
    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        # test loader has no labels — skip silently
        if isinstance(batch, (list, tuple)):
            X_batch, y_batch = batch
        else:
            raise ValueError("evaluate_model requires a labeled DataLoader.")

        X_batch = X_batch.to(device)
        logits  = model(X_batch).squeeze(1)

        all_logits.append(logits.cpu())
        all_labels.append(y_batch)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    y_probs = torch.sigmoid(all_logits).numpy()
    y_preds = (y_probs >= config.THRESHOLD).astype(int)
    y_true  = all_labels.numpy().astype(int)

    cm = confusion_matrix(y_true, y_preds)

    return {
        "f1":        f1_score(y_true,  y_preds,  zero_division=0),
        "precision": precision_score(y_true, y_preds, zero_division=0),
        "recall":    recall_score(y_true,    y_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_true,   y_probs),
        "confusion": cm,
        "report":    classification_report(y_true, y_preds,
                         target_names=["Noise (0)", "Pulsar (1)"],
                         zero_division=0),
        "y_true":    y_true,
        "y_probs":   y_probs,
        "y_preds":   y_preds,
    }


# USTAWIANIE NAJLEPSZEGO TRESHOLD +-------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds from 0.1 to 0.9 and find the one that maximises F1.
    Useful because 0.5 may not be optimal with imbalanced data.

    Returns:
        best_threshold : float
        best_f1        : float
    """
    best_f1, best_thresh = 0.0, 0.5

    for thresh in np.arange(0.1, 0.91, 0.01):
        preds = (y_probs >= thresh).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"[i] Best threshold: {best_thresh:.2f}  →  F1: {best_f1:.4f}")
    return round(best_thresh, 2), best_f1


# PRINT +-----------------------------------------------------------------------

def print_metrics(metrics: dict, split: str = "Validation") -> None:
    tn, fp, fn, tp = metrics["confusion"].ravel()

    print(f"\n{'='*50}")
    print(f"  {split} Results")
    print(f"{'='*50}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  :  {tn:>5}    {fp:>5}   (TN / FP)")
    print(f"  Actual 1  :  {fn:>5}    {tp:>5}   (FN / TP)")
    print(f"\n{metrics['report']}")