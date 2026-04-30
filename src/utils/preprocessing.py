import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import config


# Helpery


def _make_dirs() -> None:
    for d in [config.PROCESSED_DIR, config.CHECKPOINTS_DIR, config.PLOTS_DIR, config.PREDICTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    return train, test


def _report(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {name}  —  shape: {df.shape}")
    print(f"{'=' * 50}")

    missing = df.isnull().sum()
    if missing.any():
        print("\n[!] Missing values:")
        print(missing[missing > 0].to_string())
    else:
        print("\n[✓] No missing values")

    if config.TARGET_COLUMN in df.columns:
        counts = df[config.TARGET_COLUMN].value_counts().sort_index()
        ratio = counts.get(1, 0) / len(df)
        print(f"\n[i] Class distribution:\n{counts.to_string()}")
        print(f"    Pulsar ratio: {ratio:.2%}  →  pos_weight ≈ {counts.get(0, 1) / counts.get(1, 1):.2f}")


# Preprocessing


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    if config.POS_WEIGHT_OVERRIDE is not None:
        w = float(config.POS_WEIGHT_OVERRIDE)
        print(f'[i] Using manual pos_weight: {w:.4f}')
    else:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        w = n_neg / n_pos
        print(f'[i] Auto pos_weight = {w:.4f}')
    return torch.tensor([w], dtype=torch.float32)


def run_preprocessing() -> dict:
    _make_dirs()
    # 1. Zaladuj dane
    train_df, test_df = _load_raw()
    _report(train_df, "train.csv")
    _report(test_df, "test.csv")

    # 2. Rozdziel cechy / target
    feature_cols = config.FEATURE_COLUMNS

    X_raw = train_df[feature_cols].values.astype(np.float32)
    y = train_df[config.TARGET_COLUMN].values.astype(np.float32)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    # 3. Uzupelnij dane

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # 4. train/test split

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_imputed, y, test_size=config.VAL_SIZE, random_state=config.SEED, stratify=y
    )

    # 5. Skalowanie

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_imputed).astype(np.float32)

    # 6. pos_weight

    pos_weight = compute_pos_weight(y_train)

    # 7.

    joblib.dump(imputer, config.PROCESSED_DIR / "imputer.pkl")
    joblib.dump(scaler, config.PROCESSED_DIR / "scaler.pkl")
    print(f"\n[✓] Imputer and scaler saved to {config.PROCESSED_DIR}")

    # 8. podsumowanie

    print(f"\n[✓] Preprocessing complete")
    print(f"    X_train : {X_train.shape}  |  pulsars: {y_train.sum():.0f} / {len(y_train)}")
    print(f"    X_val   : {X_val.shape}    |  pulsars: {y_val.sum():.0f} / {len(y_val)}")
    print(f"    X_test  : {X_test.shape}")
    print(f"    pos_weight: {pos_weight.item():.4f}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "pos_weight": pos_weight,
        "feature_names": feature_cols,
    }


def load_and_transform_test() -> np.ndarray:
    """
    Load test.csv and apply the already-fitted imputer + scaler.
    Call this from predict.py — never refit on new data.
    """
    imputer = joblib.load(config.PROCESSED_DIR / "imputer.pkl")
    scaler = joblib.load(config.PROCESSED_DIR / "scaler.pkl")

    test_df = pd.read_csv(config.TEST_PATH)
    X_raw = test_df[config.FEATURE_COLUMNS].values.astype(np.float32)
    X = scaler.transform(imputer.transform(X_raw)).astype(np.float32)
    return X


if __name__ == "__main__":
    run_preprocessing()
