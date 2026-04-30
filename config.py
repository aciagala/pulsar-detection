from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pt"
SUBMISSION_PATH = PREDICTIONS_DIR / "submission.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------------
# Data & Preprocessing
# ---------------------------------------------------------------------------

TARGET_COLUMN = "target_class"

# Fraction of training data used for validation
VAL_SIZE = 0.2

# Features in order as they appear in the CSV
FEATURE_COLUMNS = [' Mean of the integrated profile',
                   ' Standard deviation of the integrated profile',
                   ' Excess kurtosis of the integrated profile',
                   ' Skewness of the integrated profile',
                   ' Mean of the DM-SNR curve',
                   ' Standard deviation of the DM-SNR curve',
                   ' Excess kurtosis of the DM-SNR curve',
                   ' Skewness of the DM-SNR curve']


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 2  # set to 0 if on Windows or debugging
PIN_MEMORY = True  # speeds up CPU→GPU transfer when using a GPU

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

INPUT_SIZE = len(FEATURE_COLUMNS)  # 8

# Hidden layer sizes — add/remove entries to change depth
HIDDEN_SIZES = [64, 32, 16]

OUTPUT_SIZE = 1  # single logit for binary classification

DROPOUT_RATE = 0.3
USE_BATCH_NORM = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularisation passed to Adam

NUM_EPOCHS = 100

# Early stopping: halt if val loss doesn't improve for this many epochs
PATIENCE = 10

# Classification threshold applied to sigmoid outputs during evaluation
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Class Imbalance
# ---------------------------------------------------------------------------

# Strategy: "pos_weight" (loss weighting) | "sampler" (WeightedRandomSampler)
# Both address imbalance; "pos_weight" is simpler, "sampler" changes batch composition.
IMBALANCE_STRATEGY = "pos_weight"

# pos_weight is computed automatically from the training set in preprocessing.py,
# but you can override it manually here (set to None for auto).
# Rule of thumb: pos_weight ≈ num_negative_samples / num_positive_samples
POS_WEIGHT_OVERRIDE = None

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Metrics computed and logged after every epoch on the validation set
EVAL_METRICS = ["f1", "precision", "recall", "roc_auc"]

# Primary metric used to decide whether to save a new best checkpoint
MONITOR_METRIC = "f1"  # alternatives: "roc_auc", "val_loss"
