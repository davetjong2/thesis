"""
main.py
=======
Entry point utama proyek Gait Authentication.

Cara penggunaan:
  python main.py --mode all                  # jalankan semua skenario
  python main.py --mode preprocess           # hanya preprocessing
  python main.py --mode scenario1            # skenario baseline saja
  python main.py --mode scenario2
  python main.py --mode scenario3
  python main.py --mode scenario4
  python main.py --mode scenario5
  python main.py --mode synthetic            # test dengan data sintetis (tanpa data nyata)
"""

import argparse
import logging
import numpy as np
import time
from pathlib import Path

from config import RANDOM_SEED, RESULTS_DIR, PLOTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def load_or_generate_data(use_synthetic: bool = False):
    """
    Muat data dari RAW_DIR atau bangkitkan data sintetis untuk testing.
    """
    from preprocessing import load_all_participants, generate_synthetic_data

    if use_synthetic:
        log.info("Menggunakan data sintetis (20 partisipan)...")
        return generate_synthetic_data(n_participants=20, seed=RANDOM_SEED)

    from config import RAW_DIR
    if not any(RAW_DIR.glob("*.csv")):
        log.warning(
            f"Tidak ada CSV di {RAW_DIR}.\n"
            "Gunakan --mode synthetic untuk menjalankan dengan data sintetis,\n"
            "atau letakkan CSV dari aplikasi Android di folder data/raw/"
        )
        return None

    return load_all_participants(RAW_DIR)


def run_preprocessing_only(all_data: dict):
    """Preprocessing dan simpan semua processed data."""
    from preprocessing import build_dataset, save_processed
    pids = list(all_data.keys())
    for pid in pids:
        try:
            X, y, scaler = build_dataset(all_data, target_pid=pid)
            save_processed(X, y, scaler, name=f"participant_{pid}")
        except Exception as e:
            log.warning(f"Skip {pid}: {e}")


def run_single_scenario(scenario_num: int, all_data: dict, X_all=None, y_all=None):
    """Jalankan satu skenario spesifik."""
    from preprocessing import build_dataset
    from experiments import (
        scenario1_baseline, scenario2_fl_vs_centralized,
        scenario3_noniid, scenario4_multiposition, scenario5_shap_analysis
    )
    from model import build_cnn_lstm, compile_model, train_centralized
    from federated import build_fl_experiment
    from experiments import split_data

    np.random.seed(RANDOM_SEED)
    pids = list(all_data.keys())

    # Build combined dataset jika belum ada
    if X_all is None:
        X_list, y_list = [], []
        for pid in pids:
            try:
                X_p, y_p, _ = build_dataset(all_data, target_pid=pid)
                X_list.append(X_p); y_list.append(y_p)
            except Exception as e:
                log.warning(f"Skip {pid}: {e}")
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

    if scenario_num == 1:
        return scenario1_baseline(X_all, y_all)

    elif scenario_num == 2:
        return scenario2_fl_vs_centralized(X_all, y_all, n_total_clients=len(pids))

    elif scenario_num == 3:
        return scenario3_noniid(X_all, y_all, n_total_clients=len(pids))

    elif scenario_num == 4:
        return scenario4_multiposition(all_data, target_pids=pids[:10])

    elif scenario_num == 5:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_all, y_all)

        model_c = compile_model(build_cnn_lstm())
        train_centralized(model_c, X_train, y_train, X_val, y_val, model_name="s5_central")

        server_fl = build_fl_experiment(
            X_train, y_train, X_val, y_val,
            n_clients=min(len(pids), 30),
            clients_per_round=10,
            alpha=1.0
        )
        server_fl.run(n_rounds=50, verbose=False)

        return scenario5_shap_analysis(
            model_c, server_fl.global_model, X_train, X_test
        )

    else:
        raise ValueError(f"Nomor skenario tidak valid: {scenario_num}")


def main():
    parser = argparse.ArgumentParser(
        description="Gait Authentication Research Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "preprocess", "scenario1", "scenario2",
                 "scenario3", "scenario4", "scenario5", "synthetic"],
        default="synthetic",
        help="Mode eksekusi (default: synthetic)"
    )
    args = parser.parse_args()

    log.info(f"Mode: {args.mode}")
    t_start = time.time()

    # ── Muat data ──────────────────────────────────────────────────────────────
    use_synthetic = args.mode == "synthetic"
    all_data = load_or_generate_data(use_synthetic=use_synthetic)

    if all_data is None:
        return

    n_participants = len(all_data)
    log.info(f"Jumlah partisipan: {n_participants}")

    # ── Dispatch mode ──────────────────────────────────────────────────────────
    if args.mode == "preprocess":
        run_preprocessing_only(all_data)

    elif args.mode == "all" or args.mode == "synthetic":
        from experiments import run_all_scenarios
        np.random.seed(RANDOM_SEED)
        run_all_scenarios(all_data)

    elif args.mode.startswith("scenario"):
        num = int(args.mode[-1])
        result = run_single_scenario(num, all_data)
        log.info(f"Skenario {num} selesai: {result}")

    elapsed = time.time() - t_start
    log.info(f"Total waktu eksekusi: {elapsed:.1f} detik")
    log.info(f"Hasil tersimpan di: {RESULTS_DIR}")
    log.info(f"Plot tersimpan di:  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
