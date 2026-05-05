# Gait Data Collector Tech Architecture

Below are the architectural and sequence diagrams for the Gait Data Collector application built using Kotlin, Jetpack Compose, and the Android Sensor API.

## High-Level Component Diagram

```mermaid
graph TD
    UI[MainActivity & Compose UI] --> VM[MainViewModel]
    VM --> SC[SensorCollector]
    VM --> EX[CsvExporter]
    
    SC --> SM[Android SensorManager]
    SM -- Polls at 200Hz+ --> HW[Gyroscope & Accelerometer hardware]

    EX --> MS[Android MediaStore]
    MS -- Writes to --> DL[Downloads Directory]
    
    VM --> TG[ToneGenerator]
    TG -- Beeps --> Audio[Audio Hardware]

    style UI fill:#e1f5fe,stroke:#01579b
    style VM fill:#ffe0b2,stroke:#e65100
    style SC fill:#e8f5e9,stroke:#1b5e20
    style EX fill:#f3e5f5,stroke:#4a148c
```

## Data Flow & Recording Sequence

```mermaid
sequenceDiagram
    actor User
    participant View as UI (Compose)
    participant VM as MainViewModel
    participant Audio as ToneGenerator
    participant Sensor as SensorCollector
    participant Hardware as Android System
    participant CSV as CsvExporter

    User->>View: Clicks "START BATCH"
    View->>VM: beginNextRecording()
    
    rect rgb(255, 243, 224)
    Note over VM, Audio: Countdown Phase (3s)
    loop 3 Times
        VM->>Audio: playSoundTick()
        VM->>View: Update timeRemaining
    end
    end
    
    rect rgb(224, 242, 241)
    Note over VM, Sensor: Recording Phase
    VM->>Audio: playSoundStart()
    VM->>Sensor: startRecording(batchName)
    
    loop SENSOR_DELAY_FASTEST
        Hardware-->>Sensor: onSensorChanged(Event)
        Sensor->>Sensor: Mutate lastAccel or lastGyro
        Sensor->>Sensor: append SensorRecord memory list
        Sensor-->>View: StateFlow emit (Live preview)
    end
    end
    
    VM->>Audio: playSoundStop()
    VM->>Sensor: stopRecording()
    
    User->>View: Clicks "SAVE DATA TO CSV"
    View->>VM: exportData()
    VM->>CSV: exportToCsv(participantId, memory list)
    CSV->>Hardware: MediaStore URI Insert
    Hardware-->>CSV: Output Stream
    CSV->>Hardware: Write bytes block
    CSV-->>VM: return Success Path
    VM-->>View: Reveal Saved Path UI
```

## Data Structure

```mermaid
classDiagram
    class SensorRecord {
        +String batchName
        +Long timestampMs
        +Float gyroX
        +Float gyroY
        +Float gyroZ
        +Float accelX
        +Float accelY
        +Float accelZ
    }

    class MainViewModel {
        +StateFlow batches
        +StateFlow recordingState
        +StateFlow timeRemaining
        +beginNextRecording()
        +exportData()
    }
    
    class SensorCollector {
        +StateFlow liveAccel
        +StateFlow liveGyro
        +List~SensorRecord~ recordedData
        +startListening()
        +startRecording()
    }

    MainViewModel *-- SensorRecord : owns via Collector
    SensorCollector --> SensorRecord : constructs row
```
