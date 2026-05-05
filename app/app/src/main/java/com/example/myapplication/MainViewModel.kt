package com.example.myapplication

import android.app.Application
import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID

data class BatchSetting(val name: String, val durationSec: Int)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs = application.getSharedPreferences("thesis_prefs", Context.MODE_PRIVATE)
    
    val sensorCollector = SensorCollector(application)

    private val _participantId = MutableStateFlow(generateParticipantId())
    val participantId: StateFlow<String> = _participantId.asStateFlow()

    private val _hasConsent = MutableStateFlow(false)
    val hasConsent: StateFlow<Boolean> = _hasConsent.asStateFlow()

    private val _batches = MutableStateFlow<List<BatchSetting>>(emptyList())
    val batches: StateFlow<List<BatchSetting>> = _batches.asStateFlow()

    private val _currentBatchIndex = MutableStateFlow(0)
    val currentBatchIndex: StateFlow<Int> = _currentBatchIndex.asStateFlow()

    private val _recordingState = MutableStateFlow<RecordingState>(RecordingState.Idle)
    val recordingState: StateFlow<RecordingState> = _recordingState.asStateFlow()

    private val _timeRemaining = MutableStateFlow(0)
    val timeRemaining: StateFlow<Int> = _timeRemaining.asStateFlow()

    private val _exportPath = MutableStateFlow<String?>(null)
    val exportPath: StateFlow<String?> = _exportPath.asStateFlow()

    // ToneGenerator for audio feedback (Volume at 100)
    private var toneGenerator: ToneGenerator? = null

    init {
        loadBatches()
        try {
            toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("AUDIO", "Could not initialize ToneGenerator")
        }
    }

    override fun onCleared() {
        super.onCleared()
        toneGenerator?.release()
    }

    private fun generateParticipantId(): String {
        return "ID-" + UUID.randomUUID().toString().substring(0, 6).uppercase()
    }

    fun setConsent(given: Boolean) {
        _hasConsent.value = given
    }

    fun addBatch(name: String, durationSec: Int) {
        val newBatches = _batches.value.toMutableList().apply {
            add(BatchSetting(name, durationSec))
        }
        _batches.value = newBatches
        saveBatches(newBatches)
    }

    fun removeBatch(index: Int) {
        val newBatches = _batches.value.toMutableList().apply {
            removeAt(index)
        }
        _batches.value = newBatches
        saveBatches(newBatches)
    }

    private fun saveBatches(list: List<BatchSetting>) {
        val str = list.joinToString("|") { "${it.name},${it.durationSec}" }
        prefs.edit().putString("batches_list", str).apply()
    }

    private fun loadBatches() {
        val str = prefs.getString("batches_list", "") ?: ""
        if (str.isNotBlank()) {
            _batches.value = str.split("|").map { 
                val parts = it.split(",")
                BatchSetting(parts[0], parts[1].toIntOrNull() ?: 20)
            }
        }
    }

    fun startListening() {
        sensorCollector.startListening()
    }

    fun stopListening() {
        sensorCollector.stopListening()
    }

    fun beginNextRecording() {
        if (_currentBatchIndex.value >= _batches.value.size) {
            _recordingState.value = RecordingState.Finished
            return
        }

        val currentBatch = _batches.value[_currentBatchIndex.value]
        
        viewModelScope.launch {
            _recordingState.value = RecordingState.Countdown
            _timeRemaining.value = 3
            
            // Countdown 3, 2, 1
            for (i in 3 downTo 1) {
                _timeRemaining.value = i
                playSoundTick()
                delay(1000)
            }

            // Start Recording
            playSoundStart()
            _recordingState.value = RecordingState.Recording
            _timeRemaining.value = currentBatch.durationSec
            sensorCollector.startRecording(currentBatch.name, _participantId.value)

            for (i in currentBatch.durationSec downTo 1) {
                _timeRemaining.value = i
                delay(1000)
            }

            // Stop Recording
            sensorCollector.stopRecording()
            playSoundStop()
            
            _recordingState.value = RecordingState.Idle
            _currentBatchIndex.value += 1
            
            if (_currentBatchIndex.value >= _batches.value.size) {
                _recordingState.value = RecordingState.Finished
            }
        }
    }

    fun exportData() {
        val path = CsvExporter.exportToCsv(
            getApplication(),
            _participantId.value,
            sensorCollector.recordedData
        )
        _exportPath.value = path
    }

    fun resetSession() {
        _participantId.value = generateParticipantId()
        _hasConsent.value = false
        _currentBatchIndex.value = 0
        _recordingState.value = RecordingState.Idle
        _exportPath.value = null
        sensorCollector.clearData()
    }

    private fun playSoundTick() {
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP, 100)
        } catch (e: Exception) {
            Log.e("AUDIO", "Failed to play sound: ${e.message}")
        }
    }

    private fun playSoundStart() {
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 400)
        } catch (e: Exception) {
            Log.e("AUDIO", "Failed to play sound: ${e.message}")
        }
    }

    private fun playSoundStop() {
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ABBR_ALERT, 600)
        } catch (e: Exception) {
            Log.e("AUDIO", "Failed to play sound: ${e.message}")
        }
    }
}

enum class RecordingState { Idle, Countdown, Recording, Finished }
