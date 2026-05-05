# preprocessing_ouisir.py
# =======================
# Pipeline preprocessing sinyal gait mentah OU-ISIR menjadi window tensor siap pakai.
#
# Tahapan:
#   1. Load CSV format khas OU-ISIR (2 baris header metadata, lalu data 6 kolom)
#   2. Linear interpolation untuk missing samples (grid waktu uniform)
#   3. Butterworth low-pass filter (cutoff 20 Hz, order 4)
#   4. Sliding window segmentation (128 samples, overlap 50%)
#   5. Z-score normalization per sensor axis
#   6. Build dataset: genuine vs impostor per partisipan
#
# Dua mode eksperimen:
#   - Closed-set identification : semua partisipan diklasifikasi (multiclass)
#   - Open-set verification     : genuine (1) vs impostor (0) per partisipan
#
# Referensi dataset:
#   Ngo et al., Pattern Recognition, 2014.

import re
import pickle
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

from config import (
    SUBSET_AUTO, SUBSET_MANUAL_DIRS,
    PROCESSED_DIR, PROTOCOL_DIR,
    SENSOR_AXES, N_CHANNELS, SAMPLING_RATE,
    BUTTER_CUTOFF, BUTTER_ORDER,
    WINDOW_SIZE, WINDOW_STEP,
    GAIT_STYLES_AUTO, GAIT_STYLES_MANUAL,
    GALLERY_STYLES, PROBE_STYLES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    IMPOSTOR_RATIO, RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADER — Format CSV khas OU-ISIR
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ouisir_filename(filepath: Path) -> Dict[str, str]:
    """Ekstrak metadata dari nama file OU-ISIR."""
    name = filepath.stem

    pattern_auto   = r"T0_(?:ID)?(\d+)_Center_(seq\d+)"
    pattern_manual = r"T0_(?:ID)?(\d+)_(Walk1|Walk2|SlopeUp|SlopeDown)"

    m_auto = re.match(pattern_auto, name)
    if m_auto:
        return {
            "subject_id": m_auto.group(1),
            "style"     : m_auto.group(2),
            "sensor"    : "center_auto",
        }

    m_man = re.match(pattern_manual, name)
    if m_man:
        parent = filepath.parent.name.lower()
        sensor_map = {
            "imuzleft" : "left",
            "imuzright": "right",
            "imuzcenter": "center",
            "android"  : "phone",
        }
        sensor = sensor_map.get(parent, "unknown")
        return {
            "subject_id": m_man.group(1),
            "style"     : m_man.group(2),
            "sensor"    : sensor,
        }

    log.warning(f"Nama file tidak dikenali: {filepath.name}")
    return {"subject_id": "unknown", "style": "unknown", "sensor": "unknown"}

def load_ouisir_csv(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Baca satu file CSV OU-ISIR."""
    meta = _parse_ouisir_filename(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    try:
        n_samples = int(lines[0].lower().replace("rows:", "").strip())
        n_dims    = int(lines[1].lower().replace("cols:", "").strip())
    except (ValueError, IndexError):
        raise ValueError(
            f"Format header tidak valid di {filepath.name}. "
            "Gagal membaca nilai 'rows' dan 'cols'."
        )

    if n_dims != N_CHANNELS:
        log.warning(
            f"{filepath.name}: n_dims={n_dims} != N_CHANNELS={N_CHANNELS}. "
            "Memuat tetap dengan N_CHANNELS kolom."
        )

    data_lines = lines[2:]
    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        vals = re.split(r"[,\s]+", line)
        try:
            rows.append([float(v) for v in vals[:N_CHANNELS]])
        except ValueError:
            continue

    if len(rows) != n_samples:
        log.warning(
            f"{filepath.name}: header bilang {n_samples} samples, "
            f"terbaca {len(rows)} baris data."
        )

    df = pd.DataFrame(rows, columns=SENSOR_AXES)
    df["sample_idx"] = np.arange(len(df))
    df["timestamp"]  = df["sample_idx"] / SAMPLING_RATE

    return df, meta


def load_subset_auto(
    subset_dir: Path = SUBSET_AUTO,
    max_subjects: Optional[int] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Muat Subset Otomatis (AutomaticExtractionData_IMUZCenter). 744 subjek."""
    if not subset_dir.exists():
        log.warning(
            f"Folder subset auto tidak ditemukan: {subset_dir}\n"
            "Menggunakan data sintetis sebagai fallback."
        )
        return generate_synthetic_ouisir(n_subjects=max_subjects or 20)

    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    csv_files = sorted(subset_dir.glob("T0_*_Center_seq*.csv"))

    if max_subjects:
        seen_subjects: List[str] = []
        filtered = []
        for fp in csv_files:
            m = _parse_ouisir_filename(fp)
            sid = m["subject_id"]
            if sid not in seen_subjects:
                seen_subjects.append(sid)
            if len(seen_subjects) > max_subjects:
                break
            filtered.append(fp)
        csv_files = filtered

    for fp in csv_files:
        try:
            df, meta = load_ouisir_csv(fp)
        except Exception as e:
            log.warning(f"Skip {fp.name}: {e}")
            continue

        sid   = meta["subject_id"]
        style = meta["style"]

        if sid not in all_data:
            all_data[sid] = {}
        all_data[sid][style] = df

    log.info(f"Subset auto dimuat: {len(all_data)} subjek dari {subset_dir.name}")
    return all_data


# ══════════════════════════════════════════════════════════════════════════════
# 2. INTERPOLASI LINEAR
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_missing(df: pd.DataFrame, fs: int = SAMPLING_RATE) -> pd.DataFrame:
    """Kompensasi missing samples menggunakan linear interpolation."""
    df = df.copy().reset_index(drop=True)
    max_idx = int(df["sample_idx"].max())
    idx_uniform = pd.DataFrame({"sample_idx": np.arange(0, max_idx + 1)})

    df_merged = pd.merge(idx_uniform, df[["sample_idx"] + SENSOR_AXES],
                         on="sample_idx", how="left")

    n_missing = df_merged[SENSOR_AXES].isna().sum().sum()
    if n_missing > 0:
        log.debug(f"Interpolasi {n_missing} nilai yang hilang.")
        df_merged[SENSOR_AXES] = df_merged[SENSOR_AXES].interpolate(
            method="linear", limit_direction="both"
        )

    remaining_nan = df_merged[SENSOR_AXES].isna().sum().sum()
    if remaining_nan > 0:
        df_merged[SENSOR_AXES] = df_merged[SENSOR_AXES].fillna(0.0)

    df_merged["timestamp"] = df_merged["sample_idx"] / fs
    return df_merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUTTERWORTH LOW-PASS FILTER
# ══════════════════════════════════════════════════════════════════════════════

def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = BUTTER_CUTOFF,
    fs: int = SAMPLING_RATE,
    order: int = BUTTER_ORDER,
) -> np.ndarray:
    """Butterworth low-pass filter pada semua channel."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    if normal_cutoff >= 1.0:
        return data

    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# 4. SLIDING WINDOW SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> np.ndarray:
    """Segmentasi sinyal ke dalam overlapping windows."""
    n_samples, n_channels = data.shape

    if n_samples < window_size:
        raise ValueError(
            f"Data terlalu pendek: {n_samples} < window_size={window_size}."
        )

    starts = np.arange(0, n_samples - window_size + 1, step)
    windows = np.stack([data[s : s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Z-SCORE NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_windows(
    windows: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Z-score normalization per sensor axis."""
    n_win, win_sz, n_ch = windows.shape
    flat = windows.reshape(-1, n_ch)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        flat_norm = scaler.fit_transform(flat)
    else:
        flat_norm = scaler.transform(flat)

    normalized = flat_norm.reshape(n_win, win_sz, n_ch)
    return normalized.astype(np.float32), scaler


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LENGKAP PER SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_sequence(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Pipeline lengkap: interpolasi -> filter -> window -> normalize."""
    df = interpolate_missing(df)
    raw = df[SENSOR_AXES].values.astype(np.float32)
    filtered = butter_lowpass_filter(raw)
    windows = sliding_window(filtered)
    windows, scaler = normalize_windows(windows, scaler=scaler, fit=fit_scaler)
    return windows, scaler


def preprocess_subject(
    subject_data: Dict[str, pd.DataFrame],
    styles: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Preprocess semua sequence untuk satu subjek."""
    if styles is None:
        styles = list(subject_data.keys())

    dfs = [subject_data[s] for s in styles if s in subject_data]
    if not dfs:
        raise ValueError(f"Tidak ada data untuk styles={styles}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["sample_idx"] = np.arange(len(combined_df))

    windows, scaler = preprocess_sequence(
        combined_df, scaler=scaler, fit_scaler=fit_scaler
    )
    return windows, scaler


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DATASET — Open-Set Verification (Genuine vs Impostor)
# ══════════════════════════════════════════════════════════════════════════════

def build_verification_dataset(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    target_sid: str,
    gallery_styles: Optional[List[str]] = None,
    probe_styles: Optional[List[str]] = None,
    impostor_ratio: int = IMPOSTOR_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Bangun dataset genuine (1) vs impostor (0) untuk satu subjek."""
    rng = np.random.default_rng(seed)

    if gallery_styles is None:
        gallery_styles = GALLERY_STYLES
    if probe_styles is None:
        probe_styles = PROBE_STYLES

    if target_sid not in all_data:
        raise ValueError(f"Subject '{target_sid}' tidak ditemukan.")

    subject_data = all_data[target_sid]

    # Genuine: gallery (fit scaler)
    gallery_dfs = [subject_data[s] for s in gallery_styles if s in subject_data]
    if not gallery_dfs:
        raise ValueError(f"Subject {target_sid}: tidak ada gallery data.")

    gallery_df = pd.concat(gallery_dfs, ignore_index=True)
    gallery_df["sample_idx"] = np.arange(len(gallery_df))
    genuine_gallery_windows, scaler = preprocess_sequence(
        gallery_df, scaler=None, fit_scaler=True
    )

    # Genuine: probe
    probe_dfs = [subject_data[s] for s in probe_styles if s in subject_data]
    if probe_dfs:
        probe_df = pd.concat(probe_dfs, ignore_index=True)
        probe_df["sample_idx"] = np.arange(len(probe_df))
        genuine_probe_windows, _ = preprocess_sequence(
            probe_df, scaler=scaler, fit_scaler=False
        )
        genuine_windows = np.concatenate(
            [genuine_gallery_windows, genuine_probe_windows], axis=0
        )
    else:
        genuine_windows = genuine_gallery_windows

    y_genuine = np.ones(len(genuine_windows), dtype=np.int32)
    log.info(f"Genuine [{target_sid}]: {len(genuine_windows)} windows")

    # Impostor
    impostor_windows_list = []
    for sid, sid_data in all_data.items():
        if sid == target_sid:
            continue
        try:
            imp_windows, _ = preprocess_subject(
                sid_data, styles=list(sid_data.keys()),
                scaler=scaler, fit_scaler=False,
            )
            impostor_windows_list.append(imp_windows)
        except Exception:
            continue

    if not impostor_windows_list:
        raise RuntimeError("Tidak ada data impostor.")

    all_impostor = np.concatenate(impostor_windows_list, axis=0)
    n_impostor = min(len(all_impostor), impostor_ratio * len(genuine_windows))
    imp_idx = rng.choice(len(all_impostor), size=n_impostor, replace=False)
    impostor_windows = all_impostor[imp_idx]
    y_impostor = np.zeros(n_impostor, dtype=np.int32)

    log.info(f"Impostor: {n_impostor} windows (ratio {impostor_ratio}x)")

    X = np.concatenate([genuine_windows, impostor_windows], axis=0)
    y = np.concatenate([y_genuine, y_impostor], axis=0)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm], scaler


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DATASET — Closed-Set Identification (Multiclass)
# ══════════════════════════════════════════════════════════════════════════════

def build_identification_dataset(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    gallery_styles: Optional[List[str]] = None,
    probe_styles: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], StandardScaler]:
    """Bangun dataset multiclass: setiap subjek = satu kelas."""
    rng = np.random.default_rng(seed)

    if gallery_styles is None:
        gallery_styles = GALLERY_STYLES
    if probe_styles is None:
        probe_styles = PROBE_STYLES

    sorted_sids = sorted(all_data.keys())
    label_map = {sid: idx for idx, sid in enumerate(sorted_sids)}
    log.info(f"Closed-set identification: {len(label_map)} kelas")

    # Fit global scaler on all gallery data
    all_gallery_raw = []
    for sid in sorted_sids:
        dfs = [all_data[sid][s] for s in gallery_styles if s in all_data[sid]]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        combined["sample_idx"] = np.arange(len(combined))
        combined = interpolate_missing(combined)
        raw = combined[SENSOR_AXES].values.astype(np.float32)
        filt = butter_lowpass_filter(raw)
        try:
            wins = sliding_window(filt)
            all_gallery_raw.append(wins.reshape(-1, wins.shape[-1]))
        except ValueError:
            continue

    flat_all = np.concatenate(all_gallery_raw, axis=0)
    global_scaler = StandardScaler().fit(flat_all)

    # Build gallery
    gallery_X_list, gallery_y_list = [], []
    for sid in sorted_sids:
        dfs = [all_data[sid][s] for s in gallery_styles if s in all_data[sid]]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        combined["sample_idx"] = np.arange(len(combined))
        try:
            windows, _ = preprocess_sequence(combined, scaler=global_scaler, fit_scaler=False)
            gallery_X_list.append(windows)
            gallery_y_list.append(np.full(len(windows), label_map[sid], dtype=np.int32))
        except ValueError:
            continue

    X_gallery = np.concatenate(gallery_X_list, axis=0)
    y_gallery = np.concatenate(gallery_y_list, axis=0)
    perm = rng.permutation(len(X_gallery))
    X_gallery, y_gallery = X_gallery[perm], y_gallery[perm]

    # Build probe
    probe_X_list, probe_y_list = [], []
    for sid in sorted_sids:
        dfs = [all_data[sid][s] for s in probe_styles if s in all_data[sid]]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        combined["sample_idx"] = np.arange(len(combined))
        try:
            windows, _ = preprocess_sequence(combined, scaler=global_scaler, fit_scaler=False)
            probe_X_list.append(windows)
            probe_y_list.append(np.full(len(windows), label_map[sid], dtype=np.int32))
        except ValueError:
            continue

    X_probe = np.concatenate(probe_X_list, axis=0)
    y_probe = np.concatenate(probe_y_list, axis=0)

    log.info(f"Dataset: gallery={X_gallery.shape}, probe={X_probe.shape}")
    return X_gallery, y_gallery, X_probe, y_probe, label_map, global_scaler


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(
    X: np.ndarray, y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    stratify: bool = True,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset menjadi train / val / test."""
    strat = y if stratify else None
    test_size = 1.0 - train_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=seed,
    )

    val_size_relative = val_ratio / test_size
    strat_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_size_relative),
        stratify=strat_temp, random_state=seed,
    )

    log.info(f"Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_ouisir(
    n_subjects: int = 20,
    n_seconds_per_subject: int = 30,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Buat dataset sintetis format OU-ISIR untuk testing pipeline."""
    rng = np.random.default_rng(seed)
    n_samples = n_seconds_per_subject * SAMPLING_RATE
    t = np.arange(n_samples) / SAMPLING_RATE

    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for i in range(n_subjects):
        sid = f"{i+1:06d}"
        freq_dominant = 0.8 + i * 0.1
        data[sid] = {}

        for style in GAIT_STYLES_AUTO:
            sensor_data = np.zeros((n_samples, N_CHANNELS), dtype=np.float32)
            for ch in range(N_CHANNELS):
                phase = rng.uniform(0, 2 * np.pi)
                amplitude = rng.uniform(0.5, 2.0)
                harmonic = rng.uniform(0.1, 0.4) * np.sin(
                    2 * np.pi * freq_dominant * 2 * t + rng.uniform(0, np.pi)
                )
                noise = rng.normal(0, 0.05, n_samples)
                sensor_data[:, ch] = (
                    amplitude * np.sin(2 * np.pi * freq_dominant * t + phase)
                    + harmonic + noise
                )

            df = pd.DataFrame(sensor_data, columns=SENSOR_AXES)
            df["sample_idx"] = np.arange(n_samples)
            df["timestamp"] = t
            data[sid][style] = df

    log.info(f"Synthetic OU-ISIR: {n_subjects} subjects x {len(GAIT_STYLES_AUTO)} styles")
    return data
