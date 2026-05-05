# Thesis Diagrams — Explicit Gait Verification for Secure Authentication Using Explainable Federated Learning

All diagrams below are in Mermaid format, ready to render or export as images for your thesis.

---

## 1. Kerangka Berpikir (Conceptual Framework Flowchart)

This is the end-to-end research flow from problem identification to conclusions.

```mermaid
flowchart TD
    A["Problem Statement:<br/>Conventional auth vulnerable<br/>to theft & shoulder surfing"] --> B["Research Question:<br/>Can gait biometrics provide<br/>secure implicit authentication?"]
    
    B --> C["Literature Review"]
    C --> C1["Gait Biometrics<br/>(Accelerometer + Gyroscope)"]
    C --> C2["Deep Learning<br/>(CNN-LSTM Hybrid)"]
    C --> C3["Federated Learning<br/>(FedAvg Privacy)"]
    C --> C4["Explainable AI<br/>(SHAP Values)"]
    
    C1 --> D["Data Collection<br/>Android App<br/>IMU Sensor Recording"]
    C2 --> E["Model Design<br/>CNN-LSTM Architecture"]
    C3 --> F["FL Simulation<br/>FedAvg with Non-IID"]
    C4 --> G["XAI Analysis<br/>SHAP Feature Importance"]
    
    D --> H["Preprocessing Pipeline<br/>Interpolation → Filter → Window → Normalize"]
    H --> E
    E --> I["Centralized Training<br/>(Baseline)"]
    E --> F
    
    I --> J["Evaluation<br/>Accuracy, EER, FAR, FRR, AUC"]
    F --> J
    J --> G
    
    G --> K["Hypothesis Testing"]
    K --> K1["H1: EER < 5%<br/>(One-sample t-test)"]
    K --> K2["H2: FL Degradation < 5pp<br/>(Paired t-test)"]
    K --> K3["H3: SHAP Consistency ρ ≥ 0.70<br/>(Spearman Correlation)"]
    
    K1 --> L["Conclusions &<br/>Recommendations"]
    K2 --> L
    K3 --> L

    style A fill:#ffcdd2,stroke:#c62828
    style B fill:#fff9c4,stroke:#f9a825
    style D fill:#bbdefb,stroke:#1565c0
    style E fill:#c8e6c9,stroke:#2e7d32
    style F fill:#e1bee7,stroke:#7b1fa2
    style G fill:#ffe0b2,stroke:#e65100
    style L fill:#a5d6a7,stroke:#1b5e20
```

---

## 2. System Architecture Diagram

Full system overview from mobile data collection to model deployment.

```mermaid
flowchart LR
    subgraph Mobile["📱 Android App (Gait Data Collector)"]
        M1["Sensor Manager<br/>SENSOR_DELAY_FASTEST"]
        M2["Accelerometer<br/>(TYPE_ACCELEROMETER)"]
        M3["Gyroscope<br/>(TYPE_GYROSCOPE)"]
        M4["CSV Exporter<br/>(Downloads)"]
        M2 --> M1
        M3 --> M1
        M1 --> M4
    end

    subgraph Bridge["🔄 Data Bridge"]
        B1["convert_android_csv.py<br/>Column Mapping<br/>Downsampling 500→50 Hz"]
    end

    subgraph Pipeline["🧠 ML Pipeline (gait_auth/)"]
        direction TB
        P1["preprocessing.py<br/>Interpolation → Butterworth<br/>→ Sliding Window → Z-Score"]
        P2["model.py<br/>CNN-LSTM Architecture"]
        P3["federated.py<br/>FedAvg Simulation"]
        P4["explainability.py<br/>SHAP Analysis"]
        P5["experiments.py<br/>5 Experiment Scenarios"]
        P6["statistical_analysis.py<br/>Hypothesis Testing"]
        
        P1 --> P2
        P2 --> P3
        P2 --> P5
        P3 --> P5
        P5 --> P4
        P5 --> P6
    end

    subgraph Output["📊 Output"]
        O1["Results JSON"]
        O2["SHAP Plots"]
        O3["Comparison Tables"]
        O4["TFLite Model<br/>(On-Device)"]
    end

    Mobile --> Bridge
    Bridge --> Pipeline
    Pipeline --> Output

    style Mobile fill:#e3f2fd,stroke:#1565c0
    style Bridge fill:#fff3e0,stroke:#e65100
    style Pipeline fill:#e8f5e9,stroke:#2e7d32
    style Output fill:#f3e5f5,stroke:#7b1fa2
```

---

## 3. CNN-LSTM Model Architecture

Detailed layer-by-layer neural network structure.

```mermaid
flowchart TD
    INPUT["📥 Input Layer<br/>(batch, 128, 6)<br/>128 timesteps × 6 sensor axes"]
    
    subgraph CNN["CNN Feature Extractor"]
        CONV1["Conv1D(64, kernel=3, padding=same)"]
        BN1["BatchNormalization"]
        RELU1["ReLU Activation"]
        POOL1["MaxPooling1D(pool=2)<br/>Output: (batch, 64, 64)"]
        
        CONV2["Conv1D(128, kernel=3, padding=same)"]
        BN2["BatchNormalization"]
        RELU2["ReLU Activation"]
        POOL2["MaxPooling1D(pool=2)<br/>Output: (batch, 32, 128)"]
    end
    
    DROP1["Dropout(0.3)"]
    
    subgraph RNN["LSTM Temporal Encoder"]
        LSTM1["LSTM(128, return_sequences=True)<br/>Output: (batch, 32, 128)"]
        LSTM2["LSTM(64, return_sequences=False)<br/>Output: (batch, 64)"]
    end
    
    subgraph HEAD["Classification Head"]
        DENSE1["Dense(64, ReLU)"]
        DROP2["Dropout(0.3)"]
        OUTPUT["Dense(1, Sigmoid)<br/>P(genuine) ∈ [0, 1]"]
    end

    INPUT --> CONV1 --> BN1 --> RELU1 --> POOL1
    POOL1 --> CONV2 --> BN2 --> RELU2 --> POOL2
    POOL2 --> DROP1
    DROP1 --> LSTM1 --> LSTM2
    LSTM2 --> DENSE1 --> DROP2 --> OUTPUT

    style INPUT fill:#e3f2fd,stroke:#1565c0
    style CNN fill:#fff8e1,stroke:#f9a825
    style RNN fill:#e8eaf6,stroke:#283593
    style HEAD fill:#fce4ec,stroke:#c62828
```

---

## 4. Data Preprocessing Pipeline

Signal processing from raw CSV to model-ready tensor.

```mermaid
flowchart TD
    RAW["📄 Raw CSV from Android<br/>BatchName, Timestamp,<br/>x_gyro, y_gyro, z_gyro,<br/>x_accel, y_accel, z_accel"]

    CONVERT["🔄 convert_android_csv.py<br/>• Column rename<br/>• Participant ID extraction<br/>• Downsample 500→50 Hz"]

    INTERP["1️⃣ Linear Interpolation<br/>• Reindex to uniform grid (1/50s)<br/>• merge_asof tolerance ±0.5 ms<br/>• Fill missing with linear interp"]

    FILTER["2️⃣ Butterworth Low-Pass Filter<br/>• Cutoff: 20 Hz<br/>• Order: 4<br/>• filtfilt (zero-phase)"]

    WINDOW["3️⃣ Sliding Window Segmentation<br/>• Window: 128 samples (2.56s)<br/>• Overlap: 50% (step=64)<br/>• ≈ 1–2 gait cycles per window"]

    NORM["4️⃣ Z-Score Normalization<br/>• Per sensor axis (6 channels)<br/>• fit on training set only<br/>• transform val/test with same scaler"]

    TENSOR["📦 Output Tensor<br/>(N_windows, 128, 6)<br/>float32 numpy array"]

    LABELS["🏷️ Label Assignment<br/>• Genuine (target user) → 1<br/>• Impostor (others) → 0<br/>• Balance ratio 1:3"]

    RAW --> CONVERT --> INTERP --> FILTER --> WINDOW --> NORM --> TENSOR
    NORM --> LABELS

    style RAW fill:#ffecb3,stroke:#f9a825
    style CONVERT fill:#fff3e0,stroke:#e65100
    style INTERP fill:#e3f2fd,stroke:#1565c0
    style FILTER fill:#e8f5e9,stroke:#2e7d32
    style WINDOW fill:#f3e5f5,stroke:#7b1fa2
    style NORM fill:#e0f2f1,stroke:#00695c
    style TENSOR fill:#c8e6c9,stroke:#1b5e20
    style LABELS fill:#fce4ec,stroke:#c62828
```

---

## 5. Federated Learning — FedAvg Protocol

Communication rounds between server and clients.

```mermaid
sequenceDiagram
    participant S as 🖥️ FL Server<br/>(Global Model)
    participant C1 as 📱 Client 1<br/>(Participant A)
    participant C2 as 📱 Client 2<br/>(Participant B)
    participant Ck as 📱 Client K<br/>(Participant N)

    Note over S: Initialize global model w₀

    rect rgb(232, 245, 233)
    Note over S,Ck: Communication Round t = 1, 2, ..., T
    
    S->>S: Select K random clients
    
    par Distribute Global Weights
        S->>C1: Send wₜ
        S->>C2: Send wₜ
        S->>Ck: Send wₜ
    end
    
    Note over C1: Local Training<br/>E=5 epochs<br/>SGD on local data<br/>NO raw data shared
    Note over C2: Local Training<br/>E=5 epochs
    Note over Ck: Local Training<br/>E=5 epochs
    
    par Upload Local Weights
        C1->>S: Send w₁ᵗ, n₁
        C2->>S: Send w₂ᵗ, n₂
        Ck->>S: Send wₖᵗ, nₖ
    end
    
    S->>S: FedAvg Aggregation<br/>wₜ₊₁ = Σ(nₖ/N)·wₖᵗ
    S->>S: Evaluate on val set
    S->>S: Check early stopping<br/>(patience=10)
    end

    Note over S: Export global model<br/>for SHAP analysis
```

---

## 6. SHAP Explainability Pipeline

How Explainable AI is applied to the trained model.

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        GM["Global Model<br/>(Centralized or FedAvg)"]
        BG["Background Data<br/>(100 training instances)"]
        TE["Test Data<br/>(200 instances)"]
    end

    INIT["Initialize SHAP<br/>GradientExplainer<br/>(fallback: KernelExplainer)"]

    COMPUTE["Compute SHAP Values<br/>shape: (200, 128, 6)<br/>contribution per timestep × channel"]

    subgraph AGGREGATE["Aggregation Analysis"]
        CH["Channel Importance<br/>mean|SHAP| per sensor axis<br/>(acc_x, acc_y, ..., gyr_z)"]
        TM["Temporal Importance<br/>mean|SHAP| per timestep<br/>(gait cycle phase analysis)"]
        HM["Heatmap<br/>timestep × channel<br/>(128 × 6 matrix)"]
    end

    subgraph VALIDATE["Validation H3"]
        BIO["Biomechanical Check<br/>Expected top features:<br/>acc_y, gyr_z, acc_x"]
        SPEAR["Spearman Correlation<br/>Centralized vs Federated<br/>ρ ≥ 0.70 required"]
    end

    subgraph PLOTS["Output Visualizations"]
        P1["📊 Bar Chart:<br/>Feature Importance"]
        P2["📈 Line Plot:<br/>Temporal Profile"]
        P3["🗺️ Heatmap:<br/>Timestep × Channel"]
        P4["📊 Side-by-Side:<br/>Central vs Federated"]
    end

    GM --> INIT
    BG --> INIT
    INIT --> COMPUTE
    TE --> COMPUTE
    COMPUTE --> CH
    COMPUTE --> TM
    COMPUTE --> HM
    CH --> BIO
    CH --> SPEAR
    CH --> P1
    TM --> P2
    HM --> P3
    SPEAR --> P4

    style INPUT fill:#e3f2fd,stroke:#1565c0
    style AGGREGATE fill:#fff8e1,stroke:#f9a825
    style VALIDATE fill:#fce4ec,stroke:#c62828
    style PLOTS fill:#e8f5e9,stroke:#2e7d32
```

---

## 7. Experiment Design — 5 Scenarios Mapped to Hypotheses

```mermaid
flowchart LR
    subgraph RQ["Research Questions"]
        RQ1["RQ1: Can CNN-LSTM<br/>achieve EER < 5%?"]
        RQ2["RQ2: Does FL preserve<br/>accuracy within 5pp?"]
        RQ3["RQ3: Are SHAP explanations<br/>consistent (ρ ≥ 0.70)?"]
    end

    subgraph SCENARIOS["5 Experiment Scenarios"]
        S1["S1: Centralized Baseline<br/>• Train all data pooled<br/>• Evaluate EER, Acc, AUC"]
        S2["S2: FL vs Centralized<br/>• K = 10, 25, 50<br/>• IID distribution"]
        S3["S3: Non-IID Analysis<br/>• Dirichlet α = ∞, 1.0, 0.5, 0.1<br/>• K = 10 fixed"]
        S4["S4: Multi-Position<br/>• Train: 3 positions<br/>• Test: unseen position"]
        S5["S5: SHAP Analysis<br/>• Central vs Federated<br/>• Feature ranking comparison"]
    end

    subgraph HYPO["Hypotheses"]
        H1["H1: EER < 5%<br/>One-sample t-test<br/>p < 0.05"]
        H2["H2: Degradation < 5pp<br/>Paired t-test<br/>p < 0.05"]
        H3["H3: Spearman ρ ≥ 0.70<br/>Rank correlation<br/>p < 0.05"]
    end

    RQ1 --> S1
    RQ1 --> S4
    RQ2 --> S2
    RQ2 --> S3
    RQ3 --> S5

    S1 --> H1
    S4 --> H1
    S2 --> H2
    S3 --> H2
    S5 --> H3

    style RQ fill:#e3f2fd,stroke:#1565c0
    style SCENARIOS fill:#fff8e1,stroke:#f9a825
    style HYPO fill:#c8e6c9,stroke:#2e7d32
```

---

## 8. Data Flow Diagram (DFD) — Level 1

Complete data lifecycle from collection to results.

```mermaid
flowchart TD
    USER(("👤 Participant"))
    RESEARCHER(("🧑‍🔬 Researcher<br/>(Dave Tjong)"))

    subgraph COLLECT["1.0 Data Collection"]
        DC["Android App<br/>Gait Data Collector"]
    end

    subgraph CONVERT_PROC["2.0 Data Conversion"]
        CV["convert_android_csv.py<br/>Format Mapping<br/>Downsampling"]
    end

    subgraph PREPROCESS["3.0 Preprocessing"]
        PP["preprocessing.py<br/>Interpolation<br/>Filtering<br/>Windowing<br/>Normalization"]
    end

    subgraph TRAIN["4.0 Model Training"]
        CT["Centralized<br/>Training"]
        FL["Federated<br/>Learning (FedAvg)"]
    end

    subgraph EVAL["5.0 Evaluation"]
        EV["Authentication<br/>Metrics"]
        SH["SHAP<br/>Analysis"]
        ST["Statistical<br/>Testing"]
    end

    subgraph STORE["Data Stores"]
        D1[("D1: Raw CSV<br/>data/android/")]
        D2[("D2: Processed CSV<br/>data/raw/")]
        D3[("D3: Numpy Arrays<br/>data/processed/")]
        D4[("D4: Saved Models<br/>models/saved/")]
        D5[("D5: Results<br/>results/ + plots/")]
    end

    USER -->|"Walk with phone"| DC
    RESEARCHER -->|"Configure batches"| DC
    DC -->|"CSV export"| D1
    D1 --> CV
    CV -->|"Reformatted CSV"| D2
    D2 --> PP
    PP -->|"Tensor arrays"| D3
    D3 --> CT
    D3 --> FL
    CT -->|"Model weights"| D4
    FL -->|"Global model"| D4
    D4 --> EV
    D4 --> SH
    EV -->|"Metrics JSON"| D5
    SH -->|"SHAP plots"| D5
    EV --> ST
    ST -->|"Report"| D5
    D5 --> RESEARCHER

    style USER fill:#bbdefb,stroke:#1565c0
    style RESEARCHER fill:#c8e6c9,stroke:#2e7d32
```

---

## 9. Use Case Diagram

Actor interactions with the system.

```mermaid
flowchart TD
    subgraph Actors
        P(("👤 Participant"))
        R(("🧑‍🔬 Researcher"))
    end

    subgraph System["Gait Verification System"]
        UC1["UC1: Read Consent Form<br/>(EN/ID bilingual)"]
        UC2["UC2: Give Consent<br/>(Checkbox agreement)"]
        UC3["UC3: Perform Walking Task<br/>(Follow batch instructions)"]
        UC4["UC4: Configure Batch Settings<br/>(Name, Duration)"]
        UC5["UC5: Start Recording Session<br/>(Countdown → Record → Stop)"]
        UC6["UC6: Export CSV Data<br/>(Save to Downloads)"]
        UC7["UC7: Convert CSV Format<br/>(Android → Pipeline)"]
        UC8["UC8: Run Preprocessing<br/>(Filter, Window, Normalize)"]
        UC9["UC9: Train Centralized Model"]
        UC10["UC10: Run FL Simulation<br/>(FedAvg)"]
        UC11["UC11: Run SHAP Analysis"]
        UC12["UC12: Run Experiments<br/>(5 Scenarios)"]
        UC13["UC13: Generate Reports<br/>(Hypothesis Testing)"]
        UC14["UC14: Export TFLite Model"]
    end

    P --> UC1
    P --> UC2
    P --> UC3

    R --> UC4
    R --> UC5
    R --> UC6
    R --> UC7
    R --> UC8
    R --> UC9
    R --> UC10
    R --> UC11
    R --> UC12
    R --> UC13
    R --> UC14

    UC2 -.->|"precedes"| UC3
    UC3 -.->|"precedes"| UC6
    UC6 -.->|"precedes"| UC7
    UC7 -.->|"precedes"| UC8
    UC8 -.->|"precedes"| UC9
    UC8 -.->|"precedes"| UC10
    UC9 -.->|"precedes"| UC11
    UC10 -.->|"precedes"| UC11

    style P fill:#bbdefb,stroke:#1565c0
    style R fill:#c8e6c9,stroke:#2e7d32
    style System fill:#f5f5f5,stroke:#616161
```

---

## 10. Class Diagram — Core Python Modules

```mermaid
classDiagram
    class SensorRecord {
        +String batchName
        +Long timestampMs
        +Float gyroX, gyroY, gyroZ
        +Float accelX, accelY, accelZ
    }

    class GaitCNNLSTM {
        +Input shape: (128, 6)
        +Conv1D[64, 128]
        +LSTM[128, 64]
        +Dense[64, 1]
        +build_cnn_lstm()
        +compile_model()
        +train_centralized()
        +evaluate_authentication()
    }

    class FederatedClient {
        +String client_id
        +ndarray X_local
        +ndarray y_local
        +int local_epochs
        +train(global_weights) ClientResult
    }

    class FederatedServer {
        +List~FederatedClient~ clients
        +Model global_model
        +int n_clients_per_round
        +run(n_rounds) List~RoundResult~
        -_select_clients()
        -_evaluate_global()
    }

    class NonIIDPartitioner {
        +int n_clients
        +float alpha
        +partition(X, y) List~Tuple~
    }

    class GaitSHAPAnalyzer {
        +Model model
        +ndarray X_background
        +compute_shap_values(X_test)
        +channel_importance()
        +temporal_importance()
        +validate_biomechanical_consistency()
        +run_full_analysis()
    }

    class CsvConverter {
        +convert_single_csv()
        +convert_all()
        +downsample_dataframe()
        +detect_sampling_rate()
    }

    FederatedServer *-- FederatedClient : manages
    FederatedServer --> GaitCNNLSTM : uses global model
    FederatedClient --> GaitCNNLSTM : uses local model
    NonIIDPartitioner --> FederatedServer : partitions data
    GaitSHAPAnalyzer --> GaitCNNLSTM : explains
    CsvConverter --> SensorRecord : converts from
```

---

## Quick Reference: File → Diagram Mapping

| Diagram | Source Module | Purpose |
|---------|-------------|---------|
| 1. Kerangka Berpikir | (conceptual) | Research methodology overview |
| 2. System Architecture | All modules | Technical system overview |
| 3. CNN-LSTM | `model.py` | Neural network layer details |
| 4. Preprocessing | `preprocessing.py` | Signal processing pipeline |
| 5. FedAvg Protocol | `federated.py` | FL communication rounds |
| 6. SHAP Pipeline | `explainability.py` | XAI analysis workflow |
| 7. Experiment Design | `experiments.py` | 5 scenarios → hypotheses |
| 8. DFD | All modules | Data lifecycle |
| 9. Use Case | Android + Pipeline | Actor interactions |
| 10. Class Diagram | All `.py` modules | OOP structure |
