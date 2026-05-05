"""
export_tflite.py
================
Konversi model Keras (InceptionTime) ke format TFLite untuk inferensi on-device di Android.

InceptionTime hanya menggunakan Conv1D standar, sehingga tidak memerlukan
SELECT_TF_OPS yang dibutuhkan oleh model LSTM — menghasilkan binary lebih kecil.

Cara penggunaan:
    python export_tflite.py                                       # default: model terbaik
    python export_tflite.py --model models/saved/inceptiontime_best.keras
    python export_tflite.py --quantize                            # integer quantization
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import pickle

import tensorflow as tf

from config import (
    MODELS_DIR, WINDOW_SIZE, N_CHANNELS, SENSOR_AXES,
    PROCESSED_DIR, ROOT_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EXPORT_DIR = ROOT_DIR / "models" / "tflite"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_best_model(models_dir: Path = MODELS_DIR) -> Path:
    """Cari model terbaik yang tersimpan."""
    candidates = list(models_dir.glob("*_best.h5")) + list(models_dir.glob("*.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada model tersimpan di {models_dir}.\n"
            "Jalankan training terlebih dahulu: python main.py --mode scenario1"
        )
    # Prioritas: inceptiontime, lalu centralized, lalu apapun
    for keyword in ["inceptiontime", "centralized", "s1_", "s5_"]:
        for c in candidates:
            if keyword in c.name:
                return c
    return candidates[0]


def export_tflite(
    model_path: Path,
    output_path: Path = None,
    quantize: bool = False
) -> Path:
    """
    Konversi model Keras ke TFLite.

    Args:
        model_path: path ke file .h5 atau .keras
        output_path: path output .tflite (otomatis jika None)
        quantize: jika True, gunakan dynamic range quantization

    Returns:
        Path ke file .tflite yang dihasilkan
    """
    log.info(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Convert ke TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        log.info("Menerapkan dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Untuk full integer quantization, kita butuh representative dataset
        # Dynamic range quantization sudah cukup untuk kebanyakan use case

    # InceptionTime hanya menggunakan Conv1D, BatchNorm, Dense — semua
    # didukung native oleh TFLite tanpa SELECT_TF_OPS
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    tflite_model = converter.convert()

    if output_path is None:
        suffix = "_quantized" if quantize else ""
        output_path = EXPORT_DIR / f"gait_model{suffix}.tflite"

    output_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    log.info(f"Model TFLite tersimpan: {output_path} ({size_mb:.2f} MB)")

    return output_path


def export_scaler_params(
    scaler_path: Path = None,
    output_path: Path = None
):
    """
    Ekspor parameter scaler (mean, std) ke JSON untuk digunakan di Android.
    """
    if scaler_path is None:
        # Cari scaler yang tersimpan
        candidates = list(PROCESSED_DIR.glob("*_scaler.pkl"))
        if not candidates:
            log.warning("Tidak ada scaler ditemukan. Skip export scaler.")
            return None
        scaler_path = candidates[0]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": SENSOR_AXES,
        "window_size": WINDOW_SIZE,
        "n_channels": N_CHANNELS
    }

    if output_path is None:
        output_path = EXPORT_DIR / "scaler_params.json"

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    log.info(f"Scaler params tersimpan: {output_path}")
    return output_path


def verify_tflite(tflite_path: Path):
    """Verifikasi model TFLite dengan dummy input."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    log.info(f"Input shape:  {input_details[0]['shape']}")
    log.info(f"Input dtype:  {input_details[0]['dtype']}")
    log.info(f"Output shape: {output_details[0]['shape']}")

    # Dummy inference
    dummy_input = np.random.randn(1, WINDOW_SIZE, N_CHANNELS).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    log.info(f"Dummy inference output: {output.flatten()[0]:.4f} (should be 0-1)")
    assert 0 <= output.flatten()[0] <= 1, "Output di luar range sigmoid [0, 1]!"
    log.info("✓ Verifikasi TFLite berhasil!")


def main():
    parser = argparse.ArgumentParser(description="Export model ke TFLite")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path ke model Keras (.h5 atau .keras)"
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Terapkan dynamic range quantization"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path output .tflite"
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_best_model()
    output_path = Path(args.output) if args.output else None

    tflite_path = export_tflite(model_path, output_path, args.quantize)
    export_scaler_params()
    verify_tflite(tflite_path)

    log.info(f"\nFile tersimpan di: {EXPORT_DIR}")
    log.info("Untuk integrasi Android, copy file berikut ke app/src/main/assets/:")
    log.info(f"  - {tflite_path.name}")
    log.info(f"  - scaler_params.json")


if __name__ == "__main__":
    main()
