package com.example.myapplication

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*

/**
 * Data class matching the ML pipeline expected format exactly:
 * timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, participant_id, position
 */
data class SensorRecord(
    val timestamp: Long,
    val accX: Float,
    val accY: Float,
    val accZ: Float,
    val gyrX: Float,
    val gyrY: Float,
    val gyrZ: Float,
    val participantId: String,
    val position: String
)

object CsvExporter {

    fun exportToCsv(
        context: Context,
        participantId: String,
        records: List<SensorRecord>
    ): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        val fileName = "GaitData_${participantId}_${dateFormat.format(Date())}.csv"
        
        var outputStream: OutputStream? = null
        var resultPath = ""

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = context.contentResolver
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)
                if (uri != null) {
                    outputStream = resolver.openOutputStream(uri)
                    resultPath = "Downloads/$fileName"
                }
            } else {
                val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!downloadsDir.exists()) downloadsDir.mkdirs()
                val file = File(downloadsDir, fileName)
                outputStream = FileOutputStream(file)
                resultPath = file.absolutePath
            }

            outputStream?.use { stream ->
                // Header matches ML pipeline preprocessing.py exactly
                val header = "timestamp,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,participant_id,position\n"
                stream.write(header.toByteArray())

                records.forEach { r ->
                    val line = "${r.timestamp},${r.accX},${r.accY},${r.accZ},${r.gyrX},${r.gyrY},${r.gyrZ},${r.participantId},${r.position}\n"
                    stream.write(line.toByteArray())
                }
            }
            return resultPath
        } catch (e: Exception) {
            e.printStackTrace()
            return "Error: ${e.message}"
        }
    }
}
