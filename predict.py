import torch
import numpy as np
import pandas as pd
import config
from src.utils.preprocessing import load_and_transform_test
from src.dataset.dataset import get_test_loader
from src.model.mlp import PulsarMLP
from src.training.trainer import load_best_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over the test loader.

    Returns:
        y_probs : class-1 probabilities  (np.ndarray, shape: [n_samples])
        y_preds : binary predictions at config.THRESHOLD (np.ndarray, shape: [n_samples])
    """
    model.eval()
    all_probs = []

    for X_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch).squeeze(1)
        probs   = torch.sigmoid(logits)
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = (all_probs >= config.THRESHOLD).astype(int)

    return all_probs, all_preds


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_submission(y_probs: np.ndarray, y_preds: np.ndarray) -> None:
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        "prediction":  y_preds,
        "probability": np.round(y_probs, 6),
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)

    n_pulsars = y_preds.sum()
    print(f"\n[✓] Submission saved → {config.SUBMISSION_PATH}")
    print(f"    Total samples : {len(y_preds)}")
    print(f"    Predicted pulsars (1) : {n_pulsars}  ({n_pulsars/len(y_preds):.2%})")
    print(f"    Predicted noise   (0) : {len(y_preds) - n_pulsars}")
    print(f"    Threshold used        : {config.THRESHOLD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")

    # 1. Load and transform test data using saved imputer + scaler
    print("\n[1/3] Loading and transforming test data...")
    X_test = load_and_transform_test()

    # 2. Build test DataLoader
    print("\n[2/3] Building test DataLoader...")
    test_loader = get_test_loader(X_test)
    print(f"    Test samples : {len(test_loader.dataset)}")

    # 3. Load best model and run inference
    print("\n[3/3] Running inference...")
    model    = load_best_model(PulsarMLP()).to(device)
    y_probs, y_preds = predict(model, test_loader, device)

    save_submission(y_probs, y_preds)


if __name__ == "__main__":
    main()