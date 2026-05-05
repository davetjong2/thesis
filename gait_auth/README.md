# Gait Authentication Research Pipeline
**Secure Gait-Based Authentication Using Explainable Federated Learning**

---

## Struktur Proyek

```
gait_auth/
|-- config.py                  Konfigurasi global (hyperparameter, path)
|-- preprocessing.py           Pipeline preprocessing sinyal sensor
|-- model.py                   Arsitektur CNN-LSTM + metrik autentikasi
|-- federated.py               Simulasi FL (FedAvg, NonIID partitioner)
|-- explainability.py          Analisis SHAP pada model global
|-- experiments.py             5 skenario eksperimen + plot
|-- statistical_analysis.py    Analisis statistik
|-- export_tflite.py           Export model ke TFLite untuk on-device inference
|-- main.py                    Entry point + CLI
|-- requirements.txt
|-- data/
    |-- raw/                   Letakkan CSV dari aplikasi Android langsung di sini
    |-- processed/             Output preprocessing (auto-generated)
|-- results/                   JSON hasil eksperimen + CSV + report
|-- plots/                     Visualisasi (PNG)
|-- models/
    |-- saved/                 Model Keras tersimpan
    |-- tflite/                Model TFLite untuk Android
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Workflow: Android → Model → Results

### Step 1: Kumpulkan data via Android App
Gunakan aplikasi "Gait Data Collector" untuk merekam sensor data.
Aplikasi sudah dikonfigurasi untuk menghasilkan CSV yang **langsung kompatibel**
dengan pipeline ML (sampling rate 50 Hz, format kolom identik).

### Step 2: Copy CSV ke folder data/raw/
```bash
adb pull /sdcard/Download/GaitData_*.csv data/raw/
```

CSV sudah memiliki format yang benar — **tidak perlu konversi tambahan**:
```
timestamp,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,participant_id,position
```

### Step 3: Jalankan eksperimen
```bash
python main.py --mode all         # semua skenario
python main.py --mode synthetic   # test dengan data sintetis
python main.py --mode scenario1   # per skenario
```

### Step 4: Export model ke TFLite (opsional)
```bash
python export_tflite.py
```

---

## Arsitektur Model CNN-LSTM

```
Input: (128, 6)
  → Conv1D(64) → BN → ReLU → MaxPool(2)
  → Conv1D(128) → BN → ReLU → MaxPool(2)
  → Dropout(0.3)
  → LSTM(128, return_sequences=True)
  → LSTM(64)
  → Dense(64, ReLU) → Dropout(0.3)
  → Dense(1, Sigmoid)
```

## Skenario Eksperimen

| Skenario | Deskripsi |
|----------|-----------|
| 1 | Centralized baseline |
| 2 | FL (K=10,25,50) vs Centralized, IID |
| 3 | Non-IID (alpha=inf,1.0,0.5,0.1), K=10 |
| 4 | Multi-position robustness |
| 5 | SHAP: centralized vs federated global |
