package com.example.myapplication

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Sensor collector configured at 50 Hz (SENSOR_DELAY_GAME ≈ 20ms interval)
 * to directly match the ML pipeline's SAMPLING_RATE = 50 Hz.
 * No downsampling or conversion step needed.
 */
class SensorCollector(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private var isRecording = false
    private var currentPosition = ""   // maps to "position" column in pipeline
    private var currentParticipantId = ""
    
    val recordedData = mutableListOf<SensorRecord>()

    private val _liveAccel = MutableStateFlow(FloatArray(3))
    val liveAccel: StateFlow<FloatArray> = _liveAccel.asStateFlow()

    private val _liveGyro = MutableStateFlow(FloatArray(3))
    val liveGyro: StateFlow<FloatArray> = _liveGyro.asStateFlow()

    // Keep track of the last known values for simultaneous row generation
    private var lastGyro = FloatArray(3)
    private var lastAccel = FloatArray(3)

    fun startListening() {
        // SENSOR_DELAY_GAME = ~50 Hz (20ms), matching config.py SAMPLING_RATE = 50
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stopListening() {
        sensorManager.unregisterListener(this)
    }

    fun startRecording(position: String, participantId: String) {
        currentPosition = position
        currentParticipantId = participantId
        isRecording = true
    }

    fun stopRecording() {
        isRecording = false
    }
    
    fun clearData() {
        recordedData.clear()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        val timestamp = System.currentTimeMillis()
        
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val vals = event.values.clone()
                _liveAccel.value = vals
                lastAccel = vals
                
                if (isRecording) {
                    recordedData.add(
                        SensorRecord(
                            timestamp = timestamp,
                            accX = lastAccel[0],
                            accY = lastAccel[1],
                            accZ = lastAccel[2],
                            gyrX = lastGyro[0],
                            gyrY = lastGyro[1],
                            gyrZ = lastGyro[2],
                            participantId = currentParticipantId,
                            position = currentPosition
                        )
                    )
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                val vals = event.values.clone()
                _liveGyro.value = vals
                lastGyro = vals
                
                if (isRecording) {
                    recordedData.add(
                        SensorRecord(
                            timestamp = timestamp,
                            accX = lastAccel[0],
                            accY = lastAccel[1],
                            accZ = lastAccel[2],
                            gyrX = lastGyro[0],
                            gyrY = lastGyro[1],
                            gyrZ = lastGyro[2],
                            participantId = currentParticipantId,
                            position = currentPosition
                        )
                    )
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this thesis usage
    }
}
