# IMPORTOWANIE BIBLIOTEK
from ast import Num
from pathlib import Path
from tkinter import N
from tkinter.tix import Tree
import torch


# ŚCIEŻKI +-----------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" 
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs_ninth_testing"
USED_CONFIGS_DIR = OUTPUTS_DIR / "configs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pt"
SUBMISSION_PATH = PREDICTIONS_DIR / "submission.csv"

# SEED +--------------------------------------------------------------------

#SEED = 42
SEED = 67


# PREPROCESSING DANYCH +----------------------------------------------------

TARGET_COLUMN = "target_class"

MARK_FILLED_SAMPLES = True
MARK_COLUMN_NAME = 'filled_with_median'

# UŁAMEK DANYCH TRENINGOWYCH UŻYWANY DO VALIDACJI
VAL_SIZE = 0.2

# CECHY W KOLEJNOŚCI POJAWIANIA SIĘ W EXCELU
FEATURE_COLUMNS = [' Mean of the integrated profile',
                   ' Standard deviation of the integrated profile',
                   ' Excess kurtosis of the integrated profile',
                   ' Skewness of the integrated profile',
                   ' Mean of the DM-SNR curve',
                   ' Standard deviation of the DM-SNR curve',
                   ' Excess kurtosis of the DM-SNR curve',
                   ' Skewness of the DM-SNR curve']

if( MARK_FILLED_SAMPLES ):
    FEATURE_COLUMNS.append(MARK_COLUMN_NAME)

# DATALOADER +---------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 0  # set to 0 if on Windows or debugging
PIN_MEMORY = torch.cuda.is_available()
REPEAT_SAMPLES = True #True -> próbki między batch'ami mogą się powtarzać


# ARCHITEKTURA MODELU +------------------------------------------------------

INPUT_SIZE = len(FEATURE_COLUMNS)  # ILOŚĆ CECH WEJŚCIOWYCH

HIDDEN_SIZES = [64, 32, 16]  # ROZMIARY WARSTW UKRYTYCH

OUTPUT_SIZE = 1  # ILOŚĆ CECH WYJŚCIOWYCH ( TYLKO JEDNA )

DROPOUT_RATE = 0.3 # CZĘSTOŚĆ DROPOUT'U

USE_BATCH_NORM = True # FLAGA BATCHNORM

USE_SWISH_ACTIVATION = True # FLAGA AKTYWACJI ( FALSE -> ReLU , TRUE -> Swish )


# TRENING +-------------------------------------------------------------------

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularisation passed to Adam

NUM_EPOCHS = 100 # LICZBA EPOK

PATIENCE = 20 # ILOŚĆ EPOK PRZEZ KTÓRE LOSS MOŻE SIĘ POGARSZAĆ

THRESHOLD = 0.8 # TRESHOLD POWYŻEJ KTÓREGO OBIEKT JEST UZNAWANY ZA PULSAR


# STRATEGIE ROZMIESZCZANIA DANYCH W BATCH'ACH +-------------------------------

# Strategy: "pos_weight" (loss weighting) | "sampler" (WeightedRandomSampler)
# Both address imbalance; "pos_weight" is simpler, "sampler" changes batch composition.
IMBALANCE_STRATEGY = "sampler"

# pos_weight is computed automatically from the training set in preprocessing.py,
# but you can override it manually here (set to None for auto).
# Rule of thumb: pos_weight ≈ num_negative_samples / num_positive_samples
POS_WEIGHT_OVERRIDE = None


# EWALUACJA +-----------------------------------------------------------------

# Metrics computed and logged after every epoch on the validation set
EVAL_METRICS = ["f1", "precision", "recall", "roc_auc"]

# Primary metric used to decide whether to save a new best checkpoint
MONITOR_METRIC = "f1"  # alternatives: "roc_auc", "val_loss"

# PRINT CONFIGS +-------------------------------------------------------------

def StringConfigs():
    return( f"SEED: {SEED}\n"
    f"VAL_SIZE: {VAL_SIZE}\n"
    f"FEATURE_COLUMNS: {FEATURE_COLUMNS}\n\n"
    f"TARGET_COLUMN:{TARGET_COLUMN}\n"
    f"BATCH_SIZE: {BATCH_SIZE}\n"
    f"NUM_WORKERS: {NUM_WORKERS}\n"
    f"PIN_MEMORY: {PIN_MEMORY}\n"
    f"REPEAT_SAMPLES: {REPEAT_SAMPLES}\n\n"
    f"INPUT_SIZE: {INPUT_SIZE}\n"
    f"HIDDEN_SIZES: {HIDDEN_SIZES}\n"
    f"OUTPUT_SIZE: {OUTPUT_SIZE}\n"
    f"DROPOUT_RATE: {DROPOUT_RATE}\n"
    f"USE_BATCH_NORM: {USE_BATCH_NORM}\n"
    f"USE_SWISH_ACTIVATION: {USE_SWISH_ACTIVATION}\n\n"
    f"LEARNING_RATE: {LEARNING_RATE}\n"
    f"WEIGHT_DECAY: {WEIGHT_DECAY}\n"
    f"NUM_EPOCHS: {NUM_EPOCHS}\n"
    f"PATIENCE: {PATIENCE}\n"
    f"THRESHOLD: {THRESHOLD}\n\n"
    f"IMBALANCE_STRATEGY: {IMBALANCE_STRATEGY}\n"
    f"POS_WEIGHT_OVERRIDE: {POS_WEIGHT_OVERRIDE}\n\n"
    f"EVAL_METRICS: {EVAL_METRICS}\n"
    f"MONITOR_METRIC: {MONITOR_METRIC}\n\n")

def SaveConfigs():
    USED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(USED_CONFIGS_DIR/"used_configs.txt","w") as file:
        file.write( StringConfigs());