# IMPORTOWANIE BIBLIOTEK
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# CONFIG
import config



# KLASA DATASET +------------------------------------------------------------------
# za pomocą torch.utils.data.Dataset

class PulsarDataset(Dataset):

    # INICJALIZACJA DWOMA np.ndarray WCZYWYWANYMI Z EXCELA
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]



# WEIGHTEDRANDOMSAMPLER  +-----------------------------------------------------------
# ( używany przy strategii IMBALANCESTRATEGY = "sampler" )
# za pomocą torch.utils.data.WeightedRandomSampler

def _make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    Each sample is assigned a weight inversely proportional to its class frequency.
    This makes the DataLoader draw roughly equal numbers of pulsars and non-pulsars
    per batch, regardless of the original imbalance.
    """
    class_counts = np.bincount(y.astype(int))          # LICZBA PULSARÓW I SZUMÓW
    class_weights = 1.0 / class_counts                 # PRZYPISUJE SZANSĘ NA WYLOSOWANIE PULSAROM I SZUMOM
    sample_weights = class_weights[y.astype(int)]      # PRZYPISUJE ODPOWIEDNIĄ WAGĘ KAŻDEJ PRÓBCE

    return WeightedRandomSampler( #LOSUJE PRÓBKI WEDŁUG PRAWDOPODOBIEŃSTW - NADAJE IM KOLEJNOŚĆ
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=config.REPEAT_SAMPLES, # UWAGA! replacement = True OZNACZA ŻE PRÓBKI MOGĄ SIĘ POWTARZAĆ
    )


# DATALOADER +--------------------------------------------------------------

def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Imbalance strategy is read from config.IMBALANCE_STRATEGY:
      - "pos_weight" : no special sampling; imbalance is handled in the loss function
      - "sampler"    : use WeightedRandomSampler to rebalance batches
    """
    train_dataset = PulsarDataset(X_train, y_train)
    val_dataset   = PulsarDataset(X_val,   y_val)

    # DATALOADER ŁADUJE BATCH'E WEDŁUG RÓŻNYCH STRATEGII
    # "sampler"  - DO KAŻDEGO BATCHA 
    # default    - MIESZA DANE STANDARDOWO
    if config.IMBALANCE_STRATEGY == "sampler":
        sampler = _make_sampler(y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,            # wyklucza shuffle = True
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    else:
        # "pos_weight" / default
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

    # ŁADUJE ZBIÓR WALIDACYJNY - brak mieszania danych
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    _report(train_loader, val_loader, y_train, y_val)
    return train_loader, val_loader

# ZWRACA DATALOADER Z DANYMI Z config
def get_test_loader(X_test: np.ndarray) -> DataLoader:
    """
    Build a DataLoader for the test set (no labels).
    Used by predict.py.
    """
    return DataLoader(
        PulsarDataset(X_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


# Sanity report +------------------------------------------------------------

def _report(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    y_train:      np.ndarray,
    y_val:        np.ndarray,
) -> None:
    n_train    = len(train_loader.dataset)
    n_val      = len(val_loader.dataset)
    n_batches  = len(train_loader)
    strategy   = config.IMBALANCE_STRATEGY

    print(f"\n[✓] DataLoaders ready")
    print(f"    Train : {n_train} samples  ({int(y_train.sum())} pulsars) "
          f"→ {n_batches} batches of {config.BATCH_SIZE}")
    print(f"    Val   : {n_val} samples  ({int(y_val.sum())} pulsars)")
    print(f"    Imbalance strategy : {strategy}")

    # quick shape check on one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"    Batch shape — X: {tuple(X_batch.shape)}  y: {tuple(y_batch.shape)}")


# SZYBKI TEST +------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.preprocessing import run_preprocessing

    data = run_preprocessing()
    train_loader, val_loader = get_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
    )