import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SHAPEFHR_DATA_ROOT", PROJECT_ROOT / "local_data"))
OUTPUT_ROOT = Path(os.environ.get("SHAPEFHR_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
MODEL_VERSION = "shapefhr_paper_v3"

PAPER_SEEDS: Tuple[int, ...] = (42, 2024, 3407, 777, 1234)
PRIMARY_SEED = PAPER_SEEDS[0]

SEQ_LEN = 4800
NUM_CLASSES = 2
FINAL_TOPK_PER_CLASS = 5
TOPK_STUDY = (5, 10, 15)
LOCAL_LATENT_RADIUS = 200
GLOBAL_SIGNAL_STRIDE = 1

REFM_TEST_PATH = Path(os.environ.get("SHAPEFHR_REFM_TEST", DATA_ROOT / "BabyBeatAnalyzer_TEST.ts"))
REFM_EXPECTED_TEST_NORMAL = 519
REFM_EXPECTED_TEST_ABNORMAL = 521


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    data_path: Path
    signal_key: str
    label_key: str
    shapelets_path: Path
    split_manifest: Path
    split_counts: Tuple[int, int, int]
    class_counts: Tuple[int, int]


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = SEQ_LEN
    num_classes: int = NUM_CLASSES

    d_spe: int = 128
    d_gen: int = 64
    d_stat: int = 32

    nhead: int = 8
    num_layers: int = 1
    dim_feedforward: int = 256

    shared_kernel_size: int = 8
    generic_kernels: Tuple[int, int, int] = (7, 15, 31)
    generic_downsample_stride: int = 8


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    dropout: float

    lr_scheduler_patience: int
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_min_delta: float = 0.0

    use_weighted_sampler: bool = True
    sampler_power: float = 0.50

    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    focal_alpha: float = 0.60

    train_time_augment: bool = False
    aug_noise_std: float = 0.010
    aug_mask_ratio: float = 0.003
    aug_gain_std: float = 0.010
    aug_baseline_std: float = 0.005
    aug_max_abs_z: float = 4.0

    fp_penalty_lambda: float = 0.05

    freeze_shapelets_epochs: int = 10
    shapelet_deviation_lambda: float = 0.5
    shapelet_lr_scale: float = 0.05

    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    num_workers: int = 4


PRIVATE_I = DatasetConfig(
    key="private1",
    name="Private Dataset I",
    data_path=DATA_ROOT / "processed_batch1",
    signal_key="raw_fhr",
    label_key="label",
    shapelets_path=DATA_ROOT / "shapelets_9_1.pkl",
    split_manifest=DATA_ROOT / "splits" / "private1.npz",
    split_counts=(15246, 5082, 5083),
    class_counts=(24034, 1377),
)

PRIVATE_II = DatasetConfig(
    key="private2",
    name="Private Dataset II",
    data_path=DATA_ROOT / "processed_batch3_20min",
    signal_key="raw_fhr",
    label_key="label",
    shapelets_path=DATA_ROOT / "shapelets_batch3_rawfhr_trainonly.pkl",
    split_manifest=DATA_ROOT / "splits" / "private2.npz",
    split_counts=(34223, 11408, 11408),
    class_counts=(46155, 10884),
)

DATASETS = {
    PRIVATE_I.key: PRIVATE_I,
    PRIVATE_II.key: PRIVATE_II,
}

MODEL = ModelConfig()

# Shared settings for both private sources and all neural experiments.
TRAIN_PRIVATE_I = TrainConfig(
    batch_size=128,
    epochs=300,
    lr=1e-4,
    weight_decay=1e-3,
    patience=100,
    dropout=0.30,
    lr_scheduler_patience=5,
    sampler_power=0.50,
    fp_penalty_lambda=0.05,
    freeze_shapelets_epochs=10,
)

# The paper specifies different learning rates and shapelet freeze periods.
TRAIN_PRIVATE_II = replace(TRAIN_PRIVATE_I, lr=8e-5, freeze_shapelets_epochs=5)

TRAIN_CONFIGS = {
    PRIVATE_I.key: TRAIN_PRIVATE_I,
    PRIVATE_II.key: TRAIN_PRIVATE_II,
}

# Validation threshold search used by source-domain training.
THRESHOLD_GRID_START = 0.01
THRESHOLD_GRID_STOP = 0.991
THRESHOLD_GRID_STEP = 0.005

# Robustness experiment: realistic composite artifact corruption.
ROBUSTNESS_REPEATS = 3
ROBUSTNESS_BASE_SEED = 20260909
ROBUSTNESS_COVERAGE = (0.05, 0.10)

# =============================================================================
# Visualization-only settings
# =============================================================================

VIS_TP_COUNT = 2
VIS_TN_COUNT = 2
VIS_SAMPLE_RATE_HZ = 4.0

VIS_NORMAL_SAMPLE_NAMES: Tuple[str, ...] = ()

VIS_OUTPUT_FOLDER = "visualization_rerun"

# Complete-deceleration search.
VIS_DECEL_MIN_DROP_BPM = 15.0
VIS_DECEL_MAX_DROP_BPM = 60.0
VIS_DECEL_MIN_DURATION_SEC = 20.0
VIS_DECEL_MAX_DURATION_SEC = 180.0
VIS_DECEL_MIN_SIDE_SEC = 5.0
VIS_DECEL_BASELINE_DIFF_MAX_BPM = 12.0
VIS_DECEL_RECOVERY_MARGIN_BPM = 5.0
VIS_DECEL_CONTEXT_MARGIN_SEC = 18.0
VIS_DECEL_SEARCH_RADIUS_SEC = 120.0
VIS_DECEL_SMOOTH_SEC = 2.0

# Reject obvious artifact-like abnormal candidates.
VIS_DECEL_MIN_VALID_BPM = 50.0
VIS_DECEL_MAX_VALID_BPM = 220.0
VIS_DECEL_MAX_RAW_STEP_BPM = 35.0
VIS_DECEL_MAX_SMOOTH_STEP_BPM = 8.0

VIS_DECEL_ANCHOR_TOLERANCE_SEC = 20.0
VIS_ABNORMAL_SEARCH_ALL_POSITIVES = True
