# IMPORTOWANIE BIBLIOTEK
from turtle import pos
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
#CONFIG
import config


# WCZESNE ZATRZYMYWANIE +------------------------------------------------------------

class EarlyStopping:
    """
    Stops training if the monitored metric does not improve for `patience` epochs.
    Saves the best model checkpoint automatically.

    For loss-based monitoring : lower is better  (mode="min")
    For metric-based monitoring: higher is better (mode="max")
    """

    def __init__(self, patience: int = config.PATIENCE, mode: str = "max"):
        self.patience   = patience
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.triggered  = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        print( f"porównuję poprzedni: {score} z obecnym: {self.best_score}\n")
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def step(self, score: float, model: nn.Module) -> bool:
        
        # WYWOŁYWANY W KAŻDEJ EPOCE
        # ZWRACA TRUE KIEDY TRZEBA ZATRZYMAĆ PROGRAM
        # ZAPISUJE NAJLEPSZY STAN MODELU

        if self._is_improvement(score):
            self.best_score = score
            self.counter    = 0
            _save_checkpoint(model, score)
        else:
            self.counter += 1
            print(f"    [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered


# FUNKCJE POMOCNICZE DO CHECKPOINT +---------------------------------------------------

def _save_checkpoint(model: nn.Module, score: float) -> None:
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
    print(f"    [Checkpoint] Saved  (score: {score:.4f})  →  {config.BEST_MODEL_PATH}")


def load_best_model(model: nn.Module) -> nn.Module:
    """Load the best saved weights into a model instance."""
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    print(f"[✓] Loaded best model from {config.BEST_MODEL_PATH}")
    return model


# EWALUACJA PO WSZYSTKICH BATCH'ACH +------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
) -> dict:
    """
    Run a full pass over `loader` in eval mode.
    Returns a dict with loss, f1, precision, recall, roc_auc.
    """
    model.eval()

    total_loss  = 0.0
    all_logits  = []
    all_labels  = []

    for X_batch, y_batch in loader: #iteruje po batch'ach
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch).squeeze(1)          # oblicza przewidywania modelu z batcha,   squeeze(batch,) -> (batch)
        loss   = criterion(logits, y_batch)         # oblicza loss z przewidywań z batcha

        total_loss += loss.item() * len(y_batch)
        all_logits.append(logits.cpu())
        all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits) # zbiera dane dotyczące wszystkich zeranych wyników
    all_labels = torch.cat(all_labels)
    all_probs  = torch.sigmoid(all_logits).numpy()
    all_preds  = (all_probs >= config.THRESHOLD).astype(int)
    all_labels_np = all_labels.numpy().astype(int)

    avg_loss = total_loss / len(loader.dataset)

    return {
        "loss":      avg_loss,
        "f1":        f1_score(all_labels_np,  all_preds,  zero_division=0),
        "precision": precision_score(all_labels_np, all_preds, zero_division=0),
        "recall":    recall_score(all_labels_np,    all_preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_labels_np,   all_probs),
    }


# TRAINER +---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the training loop.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, pos_weight)
        history = trainer.fit()
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        pos_weight:   torch.Tensor,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[i] Using device: {self.device}")

        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        
        print(f"MÓJ POS_WEIGHT TO = {pos_weight}\n\n\n")
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self.device) #      SPECJALNY LOSS - POŁĄCZENIE BCE LOSS I SIGMOIDA
        )                                           #    POS_WEIGHT SPRAWIA ŻE MODEL JEST BARDZIEJ KARANY ZA PRZEOCZENIE PULSARA NIŻ ZA ZŁE SKLASYFIKOWANIE SZUMU

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, # TAKIE JAKBY REGUALRYZACJA RIDGE
        )

        # Monitor F1 on val set (higher = better)
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode="max" if config.MONITOR_METRIC != "val_loss" else "min",
        )

        # History for plotting
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
            "val_precision": [],
            "val_recall": [],
            "val_roc_auc": [],
        }

    # POJEDYŃCZA EPOKA +---------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader: # ITERUJE PO BATCH'ACH
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze(1)     # OBLICZA PREDYKCJE
            loss   = self.criterion(logits, y_batch)    # OBLICZA LOSS
            loss.backward()                             # OBLICZA GRADIENTY
            self.optimizer.step()                       # AKTUALIZUJE WAGI

            total_loss += loss.item() * len(y_batch)

        return total_loss / len(self.train_loader.dataset)

    # GŁÓWNA PĘTLA TRENUJĄCA +-------------------------------------------------------

    def fit(self) -> dict:
        """
        Train for up to config.NUM_EPOCHS epochs with early stopping.
        Returns the history dict (losses + metrics per epoch).
        """
        print(f"\n{'='*55}")
        print(f"  Training  —  up to {config.NUM_EPOCHS} epochs  "
              f"(patience={config.PATIENCE})")
        print(f"  Monitor   : {config.MONITOR_METRIC}")
        print(f"{'='*55}\n")

        for epoch in range(1, config.NUM_EPOCHS + 1):

            train_loss = self._train_epoch()
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])

            self._print_epoch(epoch, train_loss, val_metrics)

            # Early stopping — monitor chosen metric
            monitor_score = (
                val_metrics[config.MONITOR_METRIC]
                if config.MONITOR_METRIC != "val_loss"
                else val_metrics["loss"]
            )

            if self.early_stopping.step(monitor_score, self.model):
                print(f"\n[!] Early stopping triggered at epoch {epoch}")
                break

        print(f"\n[✓] Training complete. Best {config.MONITOR_METRIC}: "
              f"{self.early_stopping.best_score:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _print_epoch(self, epoch: int, train_loss: float, val: dict) -> None:
        print(
            f"  Epoch {epoch:>3}/{config.NUM_EPOCHS}"
            f"  |  train_loss: {train_loss:.4f}"
            f"  |  val_loss: {val['loss']:.4f}"
            f"  |  F1: {val['f1']:.4f}"
            f"  |  Precision: {val['precision']:.4f}"
            f"  |  Recall: {val['recall']:.4f}"
            f"  |  ROC-AUC: {val['roc_auc']:.4f}"
        )