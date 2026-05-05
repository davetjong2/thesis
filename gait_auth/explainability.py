"""
explainability.py
=================
Analisis SHAP (SHapley Additive exPlanations) pada model global gait authentication.

Diterapkan pada model agregasi global FedAvg, bukan model lokal masing-masing client.

Output:
  - Feature importance ranking (per sensor axis, per timestep)
  - Temporal importance heatmap (fase gait cycle mana yang informatif)
  - Validasi konsistensi dengan ekspektasi biomekanika gait
  - Summary plot dan dependence plot
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend untuk server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

from config import (
    SENSOR_AXES, N_CHANNELS, WINDOW_SIZE, SAMPLING_RATE,
    SHAP_BACKGROUND, SHAP_TEST, TOP_N_FEATURES,
    PLOTS_DIR, RESULTS_DIR
)

log = logging.getLogger(__name__)

# Expected top features berdasarkan biomekanika gait (untuk validasi H3)
EXPECTED_TOP_FEATURES = {"acc_y", "gyr_z", "acc_x"}


# ─── SHAP Wrapper ─────────────────────────────────────────────────────────────

class GaitSHAPAnalyzer:
    """
    Analisis SHAP untuk model gait authentication berbasis CNN-LSTM.

    Menggunakan GradientExplainer karena model berbasis TensorFlow/Keras,
    yang lebih efisien dibandingkan KernelExplainer untuk deep learning.
    Jika GradientExplainer tidak tersedia, fallback ke KernelExplainer.
    """

    def __init__(self, model, X_background: np.ndarray):
        """
        Args:
            model:        model Keras (global model dari FedAvg atau centralized)
            X_background: representasi background data (100 instances dari training set)
        """
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError(
                "Package 'shap' tidak ditemukan. Install dengan: pip install shap"
            )

        self.model = model
        self.X_background = X_background[:SHAP_BACKGROUND]
        self._explainer = None
        self._shap_values = None

    def _build_explainer(self):
        """Inisialisasi SHAP explainer (lazy, hanya saat diperlukan)."""
        if self._explainer is not None:
            return

        log.info("Membangun SHAP GradientExplainer...")
        try:
            import tensorflow as tf
            self._explainer = self.shap.GradientExplainer(
                self.model, self.X_background
            )
        except Exception as e:
            log.warning(f"GradientExplainer gagal ({e}), fallback ke KernelExplainer...")

            def predict_fn(X):
                return self.model.predict(X, verbose=0).flatten()

            # Gunakan flatten background untuk KernelExplainer
            bg_flat = self.X_background.reshape(len(self.X_background), -1)
            self._explainer = self.shap.KernelExplainer(
                lambda X_flat: predict_fn(
                    X_flat.reshape(-1, WINDOW_SIZE, N_CHANNELS)
                ),
                bg_flat[:50]   # KernelExplainer lebih lambat, pakai 50 saja
            )
            self._is_kernel = True
        else:
            self._is_kernel = False

    def compute_shap_values(
        self, X_test: np.ndarray
    ) -> np.ndarray:
        """
        Hitung SHAP values untuk X_test.

        Returns:
            shap_values: (n_test, window_size, n_channels) – kontribusi setiap
                         timestep x channel terhadap prediksi
        """
        self._build_explainer()
        X_eval = X_test[:SHAP_TEST]

        log.info(f"Menghitung SHAP values untuk {len(X_eval)} instances...")

        if getattr(self, "_is_kernel", False):
            X_flat = X_eval.reshape(len(X_eval), -1)
            sv_flat = self._explainer.shap_values(X_flat)
            sv = sv_flat.reshape(len(X_eval), WINDOW_SIZE, N_CHANNELS)
        else:
            sv_list = self._explainer.shap_values(X_eval)
            # GradientExplainer mengembalikan list untuk multi-output;
            # model kita single-output sehingga ambil elemen pertama
            sv = sv_list[0] if isinstance(sv_list, list) else sv_list

        self._shap_values = sv
        log.info(f"SHAP values computed: shape={sv.shape}")
        return sv

    # ─── Aggregated Importances ───────────────────────────────────────────────

    def channel_importance(
        self, shap_values: np.ndarray
    ) -> pd.Series:
        """
        Agregasi |SHAP| per sensor axis (rata-rata atas semua instances dan timesteps).

        Returns:
            pd.Series dengan index=SENSOR_AXES, nilai=mean |SHAP|
        """
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 1))  # (n_channels,)
        return pd.Series(mean_abs, index=SENSOR_AXES).sort_values(ascending=False)

    def temporal_importance(
        self, shap_values: np.ndarray
    ) -> np.ndarray:
        """
        Agregasi |SHAP| per timestep (rata-rata atas instances dan channels).

        Returns:
            array (window_size,) – profil temporal kepentingan fitur
        """
        return np.mean(np.abs(shap_values), axis=(0, 2))   # (window_size,)

    def timestep_channel_heatmap(
        self, shap_values: np.ndarray
    ) -> np.ndarray:
        """
        Heatmap kepentingan (timestep x channel).

        Returns:
            array (window_size, n_channels)
        """
        return np.mean(np.abs(shap_values), axis=0)   # (window_size, n_channels)

    # ─── Validation H3 ────────────────────────────────────────────────────────

    def validate_biomechanical_consistency(
        self, channel_imp: pd.Series, top_n: int = 5
    ) -> Dict:
        """
        Validasi H3: apakah fitur SHAP tertinggi konsisten dengan
        ekspektasi biomekanika gait.

        Fitur yang diharapkan masuk top-5:
          acc_y  – gerakan vertikal (ground reaction force)
          gyr_z  – rotasi lateral (stride rotation)
          acc_x  – gerakan anteroposterior

        Returns:
            dict dengan hasil validasi
        """
        top_features   = set(channel_imp.head(top_n).index.tolist())
        intersection   = top_features & EXPECTED_TOP_FEATURES
        n_match        = len(intersection)
        h3_confirmed   = n_match >= 2   # minimal 2 dari 3 expected features

        result = {
            "top_features":     list(channel_imp.head(top_n).index),
            "expected_features": list(EXPECTED_TOP_FEATURES),
            "matched":          list(intersection),
            "n_match":          n_match,
            "h3_confirmed":     h3_confirmed
        }

        log.info(
            f"H3 Validation | Top-{top_n} features: {result['top_features']} | "
            f"Matched: {result['matched']} | H3 confirmed: {h3_confirmed}"
        )
        return result

    def compare_centralized_vs_federated(
        self,
        imp_centralized: pd.Series,
        imp_federated: pd.Series
    ) -> Dict:
        """
        Hitung Spearman rank correlation antara feature importance
        model centralized vs model federated global.

        Digunakan untuk mengevaluasi konsistensi explainability antar model.
        """
        from scipy.stats import spearmanr
        # Pastikan urutan axis sama
        axes = SENSOR_AXES
        rho, pval = spearmanr(
            imp_centralized[axes].values,
            imp_federated[axes].values
        )
        return {
            "spearman_rho": float(rho),
            "p_value":      float(pval),
            "consistent":   rho >= 0.70
        }

    # ─── Visualizations ───────────────────────────────────────────────────────

    def plot_channel_importance(
        self,
        channel_imp: pd.Series,
        title: str = "Feature Importance per Sensor Axis",
        save_path: Optional[Path] = None
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2196F3" if ax_name in EXPECTED_TOP_FEATURES
                  else "#90CAF9" for ax_name in channel_imp.index]
        bars = ax.bar(channel_imp.index, channel_imp.values, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Sensor Axis", fontsize=11)
        ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2196F3", label="Expected top feature"),
            Patch(facecolor="#90CAF9", label="Other feature")
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved: {save_path}")
        plt.close()

    def plot_temporal_importance(
        self,
        temporal_imp: np.ndarray,
        title: str = "Temporal Feature Importance",
        save_path: Optional[Path] = None
    ):
        time_axis = np.arange(len(temporal_imp)) / SAMPLING_RATE   # detik
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_axis, temporal_imp, color="#2196F3", linewidth=1.5)
        ax.fill_between(time_axis, temporal_imp, alpha=0.25, color="#2196F3")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Waktu (detik)", fontsize=11)
        ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
        ax.grid(linestyle="--", alpha=0.4)
        # Tandai perkiraan gait cycle (setiap ~0.9 detik pada 50 Hz)
        gait_period = 0.9
        t_max = time_axis[-1]
        for t_mark in np.arange(gait_period, t_max, gait_period):
            ax.axvline(t_mark, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(0.98, 0.92, "| = gait cycle", transform=ax.transAxes,
                ha="right", fontsize=8, color="red", alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved: {save_path}")
        plt.close()

    def plot_heatmap(
        self,
        heatmap: np.ndarray,
        title: str = "SHAP Heatmap: Timestep x Channel",
        save_path: Optional[Path] = None
    ):
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            heatmap.T,  # (channels, timesteps)
            aspect="auto",
            cmap="Blues",
            origin="lower"
        )
        ax.set_yticks(range(N_CHANNELS))
        ax.set_yticklabels(SENSOR_AXES, fontsize=10)
        n_ticks = 10
        tick_positions = np.linspace(0, WINDOW_SIZE - 1, n_ticks, dtype=int)
        tick_labels = [f"{p / SAMPLING_RATE:.2f}s" for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, fontsize=9)
        ax.set_xlabel("Waktu", fontsize=11)
        ax.set_ylabel("Sensor Axis", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, label="Mean |SHAP|")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved: {save_path}")
        plt.close()

    def run_full_analysis(
        self,
        X_test: np.ndarray,
        model_label: str = "model",
        out_dir: Path = PLOTS_DIR
    ) -> Dict:
        """
        Jalankan analisis SHAP lengkap dan simpan semua visualisasi.

        Returns:
            dict berisi semua hasil analisis
        """
        sv    = self.compute_shap_values(X_test)
        ch_imp  = self.channel_importance(sv)
        t_imp   = self.temporal_importance(sv)
        heatmap = self.timestep_channel_heatmap(sv)
        val_h3  = self.validate_biomechanical_consistency(ch_imp)

        self.plot_channel_importance(
            ch_imp, title=f"Feature Importance – {model_label}",
            save_path=out_dir / f"shap_channel_{model_label}.png"
        )
        self.plot_temporal_importance(
            t_imp, title=f"Temporal Importance – {model_label}",
            save_path=out_dir / f"shap_temporal_{model_label}.png"
        )
        self.plot_heatmap(
            heatmap, title=f"SHAP Heatmap – {model_label}",
            save_path=out_dir / f"shap_heatmap_{model_label}.png"
        )

        results = {
            "model_label":     model_label,
            "channel_importance": ch_imp.to_dict(),
            "temporal_importance_mean": float(t_imp.mean()),
            "temporal_importance_std":  float(t_imp.std()),
            "h3_validation":   val_h3,
            "shap_values_shape": list(sv.shape)
        }

        # Simpan sebagai CSV
        ch_imp_df = ch_imp.reset_index()
        ch_imp_df.columns = ["sensor_axis", "mean_abs_shap"]
        ch_imp_df.to_csv(
            RESULTS_DIR / f"shap_channel_importance_{model_label}.csv",
            index=False
        )

        return results


if __name__ == "__main__":
    log.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Dummy test dengan model dan data sintetis
    import tensorflow as tf
    from model import build_cnn_lstm, compile_model

    np.random.seed(42)
    dummy_model = compile_model(build_cnn_lstm())

    X_bg   = np.random.randn(100, WINDOW_SIZE, N_CHANNELS).astype(np.float32)
    X_test = np.random.randn(50, WINDOW_SIZE, N_CHANNELS).astype(np.float32)

    analyzer = GaitSHAPAnalyzer(dummy_model, X_bg)
    results  = analyzer.run_full_analysis(X_test, model_label="test")
    log.info(f"SHAP analysis selesai: {results['h3_validation']}")
