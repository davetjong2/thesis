"""
federated.py
============
Simulasi Federated Learning menggunakan FedAvg.

Karena penelitian ini adalah proof-of-concept, FL disimulasikan pada satu
mesin tanpa socket/jaringan nyata. Setiap participant dimodelkan sebagai
satu client yang terpisah. Data mentah tidak pernah digabungkan; setiap
client hanya berbagi parameter model (bobot) ke server.

Komponen:
  - FederatedClient : pelatihan lokal per participant
  - FederatedServer : agregasi FedAvg + distribusi model global
  - NonIIDPartitioner: partisi data dengan Dirichlet distribution
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
import copy

from config import (
    WINDOW_SIZE, N_CHANNELS, LOCAL_EPOCHS, BATCH_SIZE,
    FL_CLIENTS_CONFIGS, MAX_FL_ROUNDS, PATIENCE, LEARNING_RATE,
    DIRICHLET_ALPHAS, RANDOM_SEED
)
from models import build_inceptiontime, compile_model


def get_weights(model):
    """Extract model weights."""
    return model.get_weights()


def set_weights(model, weights):
    """Set model weights."""
    model.set_weights(weights)

log = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ClientResult:
    """Hasil dari satu round pelatihan lokal di satu client."""
    client_id:  str
    n_samples:  int                      # jumlah sample lokal
    weights:    List[np.ndarray]         # parameter model setelah local training
    train_loss: float
    train_acc:  float


@dataclass
class RoundResult:
    """Hasil dari satu communication round FL."""
    round_num:      int
    global_loss:    float                # loss model global di validation set
    global_acc:     float
    n_clients:      int
    comm_cost_mb:   float                # estimasi biaya komunikasi


# ─── Non-IID Data Partitioner ─────────────────────────────────────────────────

class NonIIDPartitioner:
    """
    Partisi dataset ke beberapa client menggunakan Dirichlet distribution.

    alpha besar (>>1) = distribusi mendekati IID
    alpha kecil (<1)  = distribusi sangat heterogen (non-IID)
    """

    def __init__(self, n_clients: int, alpha: float, seed: int = RANDOM_SEED):
        self.n_clients = n_clients
        self.alpha     = alpha
        self.rng       = np.random.default_rng(seed)

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Bagi X, y ke self.n_clients partisi dengan Dirichlet heterogeneity.

        Returns:
            list of (X_client, y_client) untuk setiap client
        """
        classes = np.unique(y)
        client_indices = [[] for _ in range(self.n_clients)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            self.rng.shuffle(cls_idx)

            # Proporsi per client mengikuti Dirichlet
            proportions = self.rng.dirichlet(
                alpha=np.ones(self.n_clients) * self.alpha
            )
            # Batasi agar tidak ada client yang mendapat 0 sample
            proportions = np.maximum(proportions, 1e-3)
            proportions /= proportions.sum()

            # Split indices
            split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)
            split_points = np.clip(split_points, 0, len(cls_idx))
            splits = np.split(cls_idx, split_points[:-1])

            for c, split in enumerate(splits):
                client_indices[c].extend(split.tolist())

        # Shuffle data di setiap client
        partitions = []
        for indices in client_indices:
            if len(indices) == 0:
                log.warning("Client dengan 0 sample – skip.")
                continue
            idx = np.array(indices)
            self.rng.shuffle(idx)
            partitions.append((X[idx], y[idx]))

        return partitions

    def log_distribution(self, partitions: List[Tuple[np.ndarray, np.ndarray]]):
        """Cetak distribusi kelas per client untuk inspeksi."""
        log.info(f"Distribusi data (alpha={self.alpha}):")
        for i, (_, y_c) in enumerate(partitions):
            counts = np.bincount(y_c, minlength=2)
            log.info(f"  Client {i:3d}: {len(y_c):5d} samples | genuine={counts[1]} | impostor={counts[0]}")


# ─── Federated Client ─────────────────────────────────────────────────────────

class FederatedClient:
    """
    Merepresentasikan satu perangkat/participant dalam FL.
    Melakukan local training dan mengembalikan model weights.
    """

    def __init__(
        self,
        client_id: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        local_epochs: int = LOCAL_EPOCHS,
        batch_size: int = BATCH_SIZE
    ):
        self.client_id    = client_id
        self.X_local      = X_local
        self.y_local      = y_local
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.model        = compile_model(build_inceptiontime())

    def train(self, global_weights: List[np.ndarray]) -> ClientResult:
        """
        Terima global weights, latih secara lokal, kembalikan updated weights.

        Tidak ada data lokal yang dikirim – hanya delta parameter model.
        """
        set_weights(self.model, global_weights)

        history = self.model.fit(
            self.X_local, self.y_local,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True
        )

        train_loss = float(np.mean(history.history["loss"][-3:]))
        train_acc  = float(np.mean(history.history["accuracy"][-3:]))

        return ClientResult(
            client_id=self.client_id,
            n_samples=len(self.X_local),
            weights=get_weights(self.model),
            train_loss=train_loss,
            train_acc=train_acc
        )


# ─── FedAvg Aggregation ───────────────────────────────────────────────────────

def fedavg_aggregate(client_results: List[ClientResult]) -> List[np.ndarray]:
    """
    Weighted averaging parameter model dari semua client yang berpartisipasi.

    Formula:
        w_global = sum_k (n_k / n_total) * w_k

    di mana n_k adalah jumlah sample lokal client k.
    """
    total_samples = sum(r.n_samples for r in client_results)
    aggregated    = None

    for result in client_results:
        weight = result.n_samples / total_samples
        if aggregated is None:
            aggregated = [w * weight for w in result.weights]
        else:
            for i, w in enumerate(result.weights):
                aggregated[i] += w * weight

    return aggregated


# ─── Federated Server ─────────────────────────────────────────────────────────

class FederatedServer:
    """
    Server FL yang mengkoordinasikan training global.

    Bertanggung jawab untuk:
      1. Memilih subset client per round
      2. Mendistribusikan global weights
      3. Mengagregasi update dari client (FedAvg)
      4. Mengevaluasi model global di validation set
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_clients_per_round: int = 10,
        seed: int = RANDOM_SEED
    ):
        self.clients              = clients
        self.X_val                = X_val
        self.y_val                = y_val
        self.n_clients_per_round  = min(n_clients_per_round, len(clients))
        self.rng                  = np.random.default_rng(seed)
        self.global_model         = compile_model(build_inceptiontime())
        self.round_history: List[RoundResult] = []

    def _select_clients(self) -> List[FederatedClient]:
        """Pilih K client secara random untuk round ini."""
        return self.rng.choice(
            self.clients,
            size=self.n_clients_per_round,
            replace=False
        ).tolist()

    def _evaluate_global(self) -> Tuple[float, float]:
        """Evaluasi model global di server-side validation set."""
        result = self.global_model.evaluate(
            self.X_val, self.y_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        return float(result[0]), float(result[1])   # loss, accuracy

    def _estimate_comm_cost(self, weights: List[np.ndarray], n_clients: int) -> float:
        """
        Estimasi biaya komunikasi dalam MB per round.
        Upload: n_clients mengirim weights ke server
        Download: server kirim global weights ke n_clients
        """
        total_params = sum(w.size for w in weights)
        bytes_per_param = 4   # float32 = 4 bytes
        mb_per_transfer = (total_params * bytes_per_param) / (1024 ** 2)
        # 1 upload + 1 download per client
        return mb_per_transfer * n_clients * 2

    def run(
        self,
        n_rounds: int = MAX_FL_ROUNDS,
        patience: int = PATIENCE,
        verbose: bool = True
    ) -> List[RoundResult]:
        """
        Jalankan FL training untuk n_rounds communication rounds.

        Args:
            n_rounds: jumlah maksimum rounds
            patience: early stopping jika val_loss tidak membaik
        """
        best_val_loss = float("inf")
        no_improve    = 0

        global_weights = get_weights(self.global_model)

        for round_num in range(1, n_rounds + 1):
            selected = self._select_clients()

            # Setiap client melakukan local training
            client_results = []
            for client in selected:
                result = client.train(global_weights)
                client_results.append(result)

            # Agregasi FedAvg
            global_weights = fedavg_aggregate(client_results)
            set_weights(self.global_model, global_weights)

            # Evaluasi global
            val_loss, val_acc = self._evaluate_global()
            comm_cost = self._estimate_comm_cost(
                global_weights, len(selected)
            )

            rr = RoundResult(
                round_num=round_num,
                global_loss=val_loss,
                global_acc=val_acc,
                n_clients=len(selected),
                comm_cost_mb=comm_cost
            )
            self.round_history.append(rr)

            if verbose:
                log.info(
                    f"Round {round_num:3d}/{n_rounds} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                    f"clients={len(selected)} | comm={comm_cost:.2f} MB"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"Early stopping di round {round_num}.")
                    break

        return self.round_history


# ─── FL Experiment Builder ────────────────────────────────────────────────────

def build_fl_experiment(
    X_all: np.ndarray,
    y_all: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_clients: int,
    clients_per_round: int,
    alpha: float,
    local_epochs: int = LOCAL_EPOCHS,
    seed: int = RANDOM_SEED
) -> FederatedServer:
    """
    Factory function: partisi data, buat clients, buat server.

    Args:
        X_all, y_all:     dataset keseluruhan (training)
        X_val, y_val:     validation set untuk evaluasi global
        n_clients:        total jumlah client (= jumlah participant)
        clients_per_round: K, jumlah client yang dipilih per round
        alpha:            Dirichlet concentration parameter
    """
    partitioner = NonIIDPartitioner(
        n_clients=n_clients, alpha=alpha, seed=seed
    )
    partitions = partitioner.partition(X_all, y_all)
    partitioner.log_distribution(partitions)

    clients = [
        FederatedClient(
            client_id=f"client_{i:03d}",
            X_local=X_c,
            y_local=y_c,
            local_epochs=local_epochs
        )
        for i, (X_c, y_c) in enumerate(partitions)
    ]

    server = FederatedServer(
        clients=clients,
        X_val=X_val,
        y_val=y_val,
        n_clients_per_round=clients_per_round,
        seed=seed
    )

    log.info(
        f"FL setup: {len(clients)} clients | "
        f"K={clients_per_round} | alpha={alpha}"
    )
    return server


if __name__ == "__main__":
    from preprocessing import generate_synthetic_data, build_dataset
    from sklearn.model_selection import train_test_split

    log.info("Test FL simulation dengan data sintetis...")
    np.random.seed(RANDOM_SEED)

    all_data = generate_synthetic_data(n_participants=15)
    pids = list(all_data.keys())
    X, y, _ = build_dataset(all_data, target_pid=pids[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED
    )

    server = build_fl_experiment(
        X_tr, y_tr, X_val, y_val,
        n_clients=10, clients_per_round=5, alpha=1.0,
        local_epochs=2
    )

    history = server.run(n_rounds=5, verbose=True)
    log.info(f"FL test selesai. Rounds: {len(history)}, "
             f"Final acc: {history[-1].global_acc:.4f}")
