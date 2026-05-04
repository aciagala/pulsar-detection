import torch
import random
import numpy as np
import config
from src.utils.preprocessing import run_preprocessing
from src.dataset.dataset import get_dataloaders
from src.model.mlp import PulsarMLP
from src.training.trainer import Trainer, load_best_model
from src.evaluation.metrics import evaluate_model, find_best_threshold, print_metrics
from src.utils.visualization import plot_all


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()

    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------
    print("\n[1/5] Preprocessing...")
    data = run_preprocessing()

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    print("\n[2/5] Building DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
    )

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    print("\n[3/5] Building model...")
    model = PulsarMLP()
    model.summary()

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    print("\n[4/5] Training...")
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        pos_weight   = data["pos_weight"],
    )
    history = trainer.fit()

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating best model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = load_best_model(PulsarMLP()).to(device)
    metrics    = evaluate_model(best_model, val_loader, device)

    print_metrics(metrics, split="Validation")

    # Threshold tuning — may improve F1 further
    best_thresh, best_f1 = find_best_threshold(
        metrics["y_true"],
        metrics["y_probs"],
    )
    if best_thresh != config.THRESHOLD:
        print(f"\n[!] Consider updating config.THRESHOLD from "
              f"{config.THRESHOLD} to {best_thresh} "
              f"(F1: {metrics['f1']:.4f} → {best_f1:.4f})")

    # Plots
    plot_all(history, metrics)

    print("\n[✓] Done. Outputs saved to:", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()