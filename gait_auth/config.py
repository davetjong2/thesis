"""
config.py
=========
Konfigurasi untuk preprocessing dataset OU-ISIR Inertial Sensor Gait Database
dan training model gait authentication.

Referensi:
  Ngo et al., "The largest inertial sensor-based gait database and performance
  evaluation of gait-based personal authentication," Pattern Recognition, 2014.
"""

from pathlib import Path

# ─── Direktori ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
RAW_DIR        = ROOT_DIR / "data" / "raw"
PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
RESULTS_DIR    = ROOT_DIR / "results"
PLOTS_DIR      = ROOT_DIR / "plots"
MODELS_DIR     = ROOT_DIR / "models" / "saved"
PROTOCOL_DIR   = ROOT_DIR / "data" / "Protocols"

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sub-folder sesuai struktur OU-ISIR
SUBSET_AUTO    = RAW_DIR / "AutomaticExtractionData_IMUZCenter"   # 744 subjek
SUBSET_MANUAL  = RAW_DIR / "ManualExtractionData"

SUBSET_MANUAL_DIRS = {
    "center": SUBSET_MANUAL / "IMUZCenter",
    "left"  : SUBSET_MANUAL / "IMUZLeft",
    "right" : SUBSET_MANUAL / "IMUZRight",
    "phone" : SUBSET_MANUAL / "Android",
}

# ─── Sensor & Sinyal ─────────────────────────────────────────────────────────
# Format kolom OU-ISIR (6 kolom):
#   Gx, Gy, Gz [rad/s] — gyroscope
#   Ax, Ay, Az [g]     — accelerometer
SENSOR_AXES    = ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
N_CHANNELS     = len(SENSOR_AXES)   # 6
SAMPLING_RATE  = 100                # Hz (OU-ISIR sensor 100 Hz)

# ─── Label Gait Style ────────────────────────────────────────────────────────
GAIT_STYLES_AUTO   = ["seq0", "seq1"]
GAIT_STYLES_MANUAL = ["Walk1", "Walk2", "SlopeUp", "SlopeDown"]
GALLERY_STYLES     = ["seq0", "Walk1"]
PROBE_STYLES       = ["seq1", "Walk2", "SlopeUp", "SlopeDown"]

# ─── Preprocessing ────────────────────────────────────────────────────────────
BUTTER_CUTOFF  = 20     # Hz — cutoff low-pass filter
BUTTER_ORDER   = 4
WINDOW_SIZE    = 128    # samples — 1.28 detik @ 100 Hz (≈ 1 siklus langkah)
WINDOW_STEP    = 64     # overlap 50%

# ─── Pembagian Data ──────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.10
TEST_RATIO     = 0.20
IMPOSTOR_RATIO = 3
RANDOM_SEED    = 42

# ─── Model Input Shape ───────────────────────────────────────────────────────
INPUT_SHAPE    = (WINDOW_SIZE, N_CHANNELS)   # (128, 6)

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
MAX_EPOCHS     = 100
PATIENCE       = 15     # early stopping

# ─── Federated Learning ──────────────────────────────────────────────────────
LOCAL_EPOCHS         = 5
MAX_FL_ROUNDS        = 100
FL_CLIENTS_CONFIGS   = [10, 25, 50]
DIRICHLET_ALPHAS     = [1e9, 1.0, 0.5, 0.1]

# ─── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_BACKGROUND = 100
SHAP_TEST       = 200
