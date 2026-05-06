# IMPORTOWANIE BIBLIOTEK
import torch
import numpy as np
import pandas as pd
# CONFIG
import config

# UŻYCIE FUNKCJI ZDEFINIOWANYCH PRZEZ NAS
from src.utils.preprocessing import load_and_transform_test
from src.dataset.dataset import get_test_loader
from src.model.mlp import PulsarMLP
from src.training.trainer import load_best_model


# INFERENCJA (WNIOSKOWANIE) +---------------------------------------------------------

@torch.no_grad() # WYWOŁUJ W BLOKU no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:

    model.eval() # przełącz w tryb ewaluacji
    all_probs = []

    for X_batch in loader: # pozyskuje predykcje z wszystkich batch'y
        X_batch = X_batch.to(device)
        logits  = model(X_batch).squeeze(1)
        probs   = torch.sigmoid(logits)
        all_probs.append(probs.cpu())

    # all_probs przechowuje prawdopodobieństwa
    # all_preds przechowuje przewidywane klasy obiektów ( 0 / 1 )

    all_probs = torch.cat(all_probs).numpy() # łączy wyniki z wszystkich batch'y
    all_preds = (all_probs >= config.THRESHOLD).astype(int)

    return all_probs, all_preds # zwraca {np.ndarray , np.ndarray}


# ZAPISZ WYNIKI +-------------------------------------------------------------------
# za pomocą 

def save_submission(y_probs: np.ndarray, y_preds: np.ndarray) -> None:

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True) # tworzy folder na predykcje

    submission = pd.DataFrame({ # przeksztaca {np.ndarray , np.ndarray} w dataframe
        "prediction":  y_preds,
        "probability": np.round(y_probs, 6),
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False) # zapisuje predykcje w excelu

    n_pulsars = y_preds.sum()
    print(f"\n[✓] Submission saved → {config.SUBMISSION_PATH}")
    print(f"    Total samples : {len(y_preds)}")
    print(f"    Predicted pulsars (1) : {n_pulsars}  ({n_pulsars/len(y_preds):.2%})")
    print(f"    Predicted noise   (0) : {len(y_preds) - n_pulsars}")
    print(f"    Threshold used        : {config.THRESHOLD}")


# MAIN +---------------------------------------------------------------------+ MAIN #

def main() -> None:

    # próba ustawienia urządzenia na gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")

    # 1. ZAŁADUJ I PRZETRANSFORMUJ DANE
    # imputer ->    scaler -> 
    # za pomocą load_and_transform_test() z src/utils/preprocessing

    print("\n[1/3] Loading and transforming test data...")
    X_test = load_and_transform_test()

    # 2. ZBUDUJ DATALOADER TESTU
    # za pomocą get_test_loader() z src/dataset/dataset

    print("\n[2/3] Building test DataLoader...")
    test_loader = get_test_loader(X_test)
    print(f"    Test samples : {len(test_loader.dataset)}")

    # 3. ZAŁADUJ NAJLEPSZY MODEL I PRZEPROWADŹ INFERENCJĘ
    # za pomocą load_best_model z src/training/trainer
    # i PulsarMLP z src/model/mlp

    print("\n[3/3] Running inference...")
    model    = load_best_model(PulsarMLP()).to(device)
    y_probs, y_preds = predict(model, test_loader, device)

    save_submission(y_probs, y_preds) #zapisz w excelu


if __name__ == "__main__":
    main()