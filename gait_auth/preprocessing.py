"""
preprocessing.py
================
Pipeline preprocessing sinyal gait mentah (CSV) menjadi window tensor siap pakai.

Tahapan:
  1. Load CSV hasil aplikasi Android
  2. Linear interpolation untuk missing samples
  3. Butterworth low-pass filter (cutoff 20 Hz)
  4. Sliding window segmentation (128 samples, overlap 50%)
  5. Z-score normalization per sensor axis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import pickle
import logging

from config import (
    RAW_DIR, PROCESSED_DIR, SAMPLING_RATE, SENSOR_AXES,
    BUTTER_CUTOFF, BUTTER_ORDER, WINDOW_SIZE, WINDOW_STEP,
    POSITIONS, N_CHANNELS
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── 1. CSV Loader ────────────────────────────────────────────────────────────

def load_participant_csv(filepath: Path) -> pd.DataFrame:
    """
    Muat satu file CSV dari aplikasi Android.

    Format CSV yang diharapkan:
        timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, participant_id, position
    """
    df = pd.read_csv(filepath)
    required = ["timestamp"] + SENSOR_AXES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di {filepath.name}: {missing}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_all_participants(raw_dir: Path = RAW_DIR) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Muat semua CSV dan kelompokkan per participant dan per posisi.

    Returns:
        {
          "P001": {"pocket_left": df, "pocket_right": df, ...},
          "P002": {...},
          ...
        }
    """
    data = {}
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        log.warning(f"Tidak ada CSV di {raw_dir}. Gunakan generate_synthetic_data() untuk testing.")
        return data

    for fp in csv_files:
        df = load_participant_csv(fp)
        pid = df["participant_id"].iloc[0]
        pos = df["position"].iloc[0]
        if pid not in data:
            data[pid] = {}
        data[pid][pos] = df
        log.info(f"Loaded: {fp.name} | participant={pid} | position={pos} | rows={len(df)}")
    return data


# ─── 2. Interpolation ─────────────────────────────────────────────────────────

def interpolate_missing(df: pd.DataFrame, fs: int = SAMPLING_RATE) -> pd.DataFrame:
    """
    Linear interpolation untuk mengompensasi missing samples.

    Reindex ke grid waktu yang uniform (1/fs interval), lalu interpolasi
    nilai NaN yang muncul di posisi sample yang hilang.
    """
    df = df.copy()
    t_start = df["timestamp"].iloc[0]
    t_end   = df["timestamp"].iloc[-1]
    dt      = 1.0 / fs * 1000   # milidetik

    t_uniform = np.arange(t_start, t_end + dt, dt)
    df_uniform = pd.DataFrame({"timestamp": t_uniform})
    df_merged  = pd.merge_asof(
        df_uniform, df, on="timestamp", direction="nearest", tolerance=dt * 0.5
    )

    for col in SENSOR_AXES:
        df_merged[col] = df_merged[col].interpolate(method="linear", limit_direction="both")

    n_missing = df_merged[SENSOR_AXES].isna().sum().sum()
    if n_missing > 0:
        log.warning(f"Masih ada {n_missing} NaN setelah interpolasi – diisi 0.")
        df_merged[SENSOR_AXES] = df_merged[SENSOR_AXES].fillna(0)

    return df_merged


# ─── 3. Butterworth Low-Pass Filter ──────────────────────────────────────────

def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = BUTTER_CUTOFF,
    fs: int = SAMPLING_RATE,
    order: int = BUTTER_ORDER
) -> np.ndarray:
    """
    Terapkan Butterworth low-pass filter pada sinyal.

    Args:
        data: array shape (n_samples, n_channels)
    Returns:
        array shape yang sama, sudah difilter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return filtered


# ─── 4. Sliding Window Segmentation ──────────────────────────────────────────

def sliding_window(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP
) -> np.ndarray:
    """
    Segmentasi sinyal menjadi overlapping windows.

    Args:
        data: array shape (n_samples, n_channels)
    Returns:
        array shape (n_windows, window_size, n_channels)
    """
    n_samples, n_channels = data.shape
    windows = []
    start = 0
    while start + window_size <= n_samples:
        windows.append(data[start : start + window_size])
        start += step

    if not windows:
        raise ValueError(
            f"Data terlalu pendek untuk segmentasi: {n_samples} samples, "
            f"window_size={window_size}"
        )
    return np.array(windows)   # (n_windows, window_size, n_channels)


# ─── 5. Z-Score Normalization ─────────────────────────────────────────────────

def normalize_windows(
    windows: np.ndarray,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score normalization per channel (sensor axis).

    Scaler di-fit hanya pada training data, lalu diterapkan (transform) ke
    val/test data menggunakan scaler yang sama.

    Args:
        windows: (n_windows, window_size, n_channels)
        scaler:  StandardScaler yang sudah di-fit (untuk val/test)
        fit:     True jika ini training data
    Returns:
        normalized windows, scaler
    """
    n_win, win_sz, n_ch = windows.shape
    flat = windows.reshape(-1, n_ch)   # (n_win * win_sz, n_ch)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        flat_norm = scaler.fit_transform(flat)
    else:
        flat_norm = scaler.transform(flat)

    return flat_norm.reshape(n_win, win_sz, n_ch), scaler


# ─── Full Pipeline Per Participant ────────────────────────────────────────────

def preprocess_participant(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Jalankan pipeline lengkap untuk satu DataFrame participant.

    Returns:
        windows: (n_windows, 128, 6)
        scaler:  StandardScaler yang digunakan
    """
    df = interpolate_missing(df)
    raw = df[SENSOR_AXES].values.astype(np.float32)
    filtered = butter_lowpass_filter(raw)
    windows  = sliding_window(filtered)
    windows, scaler = normalize_windows(windows, scaler=scaler, fit=fit_scaler)
    return windows.astype(np.float32), scaler


# ─── Build Dataset ────────────────────────────────────────────────────────────

def build_dataset(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    target_pid: str,
    positions: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Bangun dataset untuk satu participant sebagai "genuine" (label=1),
    semua participant lain sebagai "impostor" (label=0).

    Args:
        all_data:    output dari load_all_participants()
        target_pid:  participant ID yang dianggap genuine
        positions:   subset posisi yang digunakan (None = semua)

    Returns:
        X: (N, 128, 6)
        y: (N,) – 0 atau 1
        scaler: StandardScaler dari training data genuine
    """
    if positions is None:
        positions = POSITIONS

    genuine_dfs = []
    for pos in positions:
        if pos in all_data.get(target_pid, {}):
            genuine_dfs.append(all_data[target_pid][pos])

    if not genuine_dfs:
        raise ValueError(f"Tidak ada data genuine untuk participant {target_pid}")

    genuine_df = pd.concat(genuine_dfs, ignore_index=True)
    genuine_windows, scaler = preprocess_participant(genuine_df, fit_scaler=True)
    y_genuine = np.ones(len(genuine_windows), dtype=np.int32)

    impostor_windows_list = []
    for pid, pos_data in all_data.items():
        if pid == target_pid:
            continue
        for pos in positions:
            if pos in pos_data:
                try:
                    imp_w, _ = preprocess_participant(pos_data[pos], scaler=scaler, fit_scaler=False)
                    impostor_windows_list.append(imp_w)
                except Exception as e:
                    log.warning(f"Skip impostor {pid}/{pos}: {e}")

    if not impostor_windows_list:
        raise ValueError("Tidak ada data impostor – cek dataset.")

    impostor_windows = np.concatenate(impostor_windows_list, axis=0)

    # Balance: ambil jumlah impostor = 3x genuine agar tidak terlalu imbalanced
    n_genuine = len(genuine_windows)
    n_impostor = min(len(impostor_windows), 3 * n_genuine)
    idx = np.random.choice(len(impostor_windows), n_impostor, replace=False)
    impostor_windows = impostor_windows[idx]
    y_impostor = np.zeros(n_impostor, dtype=np.int32)

    X = np.concatenate([genuine_windows, impostor_windows], axis=0)
    y = np.concatenate([y_genuine, y_impostor], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm], scaler


# ─── Save / Load Processed Data ───────────────────────────────────────────────

def save_processed(X, y, scaler, name: str, out_dir: Path = PROCESSED_DIR):
    np.save(out_dir / f"{name}_X.npy", X)
    np.save(out_dir / f"{name}_y.npy", y)
    with open(out_dir / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Saved processed data: {name} | X={X.shape} | y distribution={np.bincount(y)}")


def load_processed(name: str, proc_dir: Path = PROCESSED_DIR):
    X = np.load(proc_dir / f"{name}_X.npy")
    y = np.load(proc_dir / f"{name}_y.npy")
    with open(proc_dir / f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return X, y, scaler


# ─── Synthetic Data Generator (untuk testing tanpa data nyata) ───────────────

def generate_synthetic_data(
    n_participants: int = 20,
    n_genuine_windows: int = 100,
    seed: int = 42
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Buat dataset sintetis untuk keperluan pengujian kode tanpa data nyata.

    Setiap participant memiliki "signature" gait yang unik berupa pola sinusoidal
    dengan frekuensi dominan yang berbeda, ditambah noise Gaussian.
    """
    rng = np.random.default_rng(seed)
    data = {}
    t = np.arange(0, (n_genuine_windows * WINDOW_STEP + WINDOW_SIZE) / SAMPLING_RATE,
                  1 / SAMPLING_RATE)

    for i in range(n_participants):
        pid = f"P{i+1:03d}"
        data[pid] = {}
        freq_signature = 0.8 + i * 0.05   # frekuensi dominan unik per participant

        for pos in POSITIONS:
            n_samples = len(t)
            sensor_data = np.zeros((n_samples, N_CHANNELS))

            for ch in range(N_CHANNELS):
                phase = rng.uniform(0, 2 * np.pi)
                amplitude = rng.uniform(0.5, 2.0)
                noise = rng.normal(0, 0.1, n_samples)
                sensor_data[:, ch] = (
                    amplitude * np.sin(2 * np.pi * freq_signature * t + phase) + noise
                )

            df = pd.DataFrame(sensor_data, columns=SENSOR_AXES)
            df["timestamp"] = (t * 1000).astype(int)   # ms
            df["participant_id"] = pid
            df["position"] = pos
            data[pid][pos] = df

    log.info(f"Generated synthetic data: {n_participants} participants x {len(POSITIONS)} positions")
    return data


if __name__ == "__main__":
    import random
    np.random.seed(42)

    log.info("Menjalankan pipeline preprocessing dengan data sintetis...")
    all_data = generate_synthetic_data(n_participants=10)

    pids = list(all_data.keys())
    X, y, scaler = build_dataset(all_data, target_pid=pids[0])
    log.info(f"Dataset untuk {pids[0]}: X={X.shape}, y={np.bincount(y)}")

    save_processed(X, y, scaler, name=f"participant_{pids[0]}")
    log.info("Preprocessing selesai.")
