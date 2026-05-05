package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.myapplication.ui.theme.MyApplicationTheme

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppNavigation(viewModel)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.startListening()
    }

    override fun onPause() {
        super.onPause()
    }
}

@Composable
fun AppNavigation(viewModel: MainViewModel) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "home") {
        composable("home") { HomeScreen(viewModel, navController) }
        composable("settings") { SettingsScreen(viewModel, navController) }
        composable("recording") { RecordingScreen(viewModel, navController) }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(viewModel: MainViewModel, navController: NavController) {
    val participantId by viewModel.participantId.collectAsStateWithLifecycle()
    val hasConsent by viewModel.hasConsent.collectAsStateWithLifecycle()
    val batches by viewModel.batches.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Gait Verification Settings", fontWeight = FontWeight.SemiBold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                ),
                navigationIcon = {
                    IconButton(onClick = { navController.navigate("settings") }) {
                        Icon(Icons.Filled.Settings, contentDescription = "Settings")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Participant ID",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = participantId,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = "This ID is anonymous and strictly generated for GDPR compliance. Names are not tracked.",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 4.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
                    )
                }
            }

            Text("Informed Consent / Persetujuan Data", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text(
                        text = "English",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "My name is Dave Tjong, a Computer Science student at Bina Nusantara University (Student ID: 2602077736). I am collecting accelerometer and gyroscope data for my thesis titled Explicit Gait Verification for Secure Authentication Using Explainable Federated Learning.\n\nBy proceeding, you consent to the anonymous collection and use of your sensor data for academic research purposes. Participation is voluntary and you may withdraw at any time.",
                        style = MaterialTheme.typography.bodyMedium,
                        lineHeight = 22.sp
                    )

                    HorizontalDivider()

                    Text(
                        text = "Indonesian",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Nama saya Dave Tjong, mahasiswa Teknik Informatika Universitas Bina Nusantara (NIM: 2602077736). Saya sedang mengumpulkan data akselerometer dan giroskop untuk skripsi berjudul Explicit Gait Verification for Secure Authentication Using Explainable Federated Learning.\n\nDengan melanjutkan, Anda menyetujui pengumpulan dan penggunaan data sensor Anda secara anonim untuk tujuan penelitian akademis. Partisipasi bersifat sukarela dan Anda dapat menarik diri kapan saja.",
                        style = MaterialTheme.typography.bodyMedium,
                        lineHeight = 22.sp
                    )
                }
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (hasConsent) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface
                ),
                border = if (!hasConsent) null else null
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(
                        checked = hasConsent,
                        onCheckedChange = { viewModel.setConsent(it) }
                    )
                    Text(
                        text = "I consent to the collection / Saya setuju",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = if (hasConsent) FontWeight.Bold else FontWeight.Normal,
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            if (batches.isEmpty()) {
                Text(
                    text = "⚠️ Please configure recording batches in Settings first.",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }

            Button(
                onClick = {
                    navController.navigate("recording")
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp),
                enabled = hasConsent && batches.isNotEmpty()
            ) {
                Text(
                    text = "BEGIN RECORDING",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: MainViewModel, navController: NavController) {
    val batches by viewModel.batches.collectAsStateWithLifecycle()
    var newName by remember { mutableStateOf("") }
    var newDuration by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("App Settings") },
                navigationIcon = {
                    Button(
                        onClick = { navController.popBackStack() },
                        modifier = Modifier.padding(start = 8.dp),
                        colors = ButtonDefaults.textButtonColors()
                    ) {
                        Text("Back", color = MaterialTheme.colorScheme.onSurface)
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
        ) {
            Card(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text("Add Testing Batch", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    OutlinedTextField(
                        value = newName,
                        onValueChange = { newName = it },
                        label = { Text("Batch Layout (e.g., Fast Walk)") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )
                    OutlinedTextField(
                        value = newDuration,
                        onValueChange = { newDuration = it },
                        label = { Text("Duration (Seconds)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )

                    Button(
                        onClick = {
                            val duration = newDuration.toIntOrNull()
                            if (newName.isNotBlank() && duration != null) {
                                viewModel.addBatch(newName, duration)
                                newName = ""
                                newDuration = ""
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(50.dp)
                    ) {
                        Text("ADD BATCH")
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Active Batches Queue (${batches.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(horizontal = 16.dp)
            )

            LazyColumn(modifier = Modifier
                .weight(1f)
                .padding(16.dp)) {
                itemsIndexed(batches) { index, batch ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(batch.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                Text("${batch.durationSec} seconds", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            IconButton(
                                onClick = { viewModel.removeBatch(index) },
                                modifier = Modifier.background(Color.Red.copy(alpha = 0.1f), RoundedCornerShape(8.dp))
                            ) {
                                Icon(Icons.Filled.Delete, contentDescription = "Delete", tint = Color.Red)
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordingScreen(viewModel: MainViewModel, navController: NavController) {
    val participantId by viewModel.participantId.collectAsStateWithLifecycle()
    val batches by viewModel.batches.collectAsStateWithLifecycle()
    val currentIndex by viewModel.currentBatchIndex.collectAsStateWithLifecycle()
    val recordingState by viewModel.recordingState.collectAsStateWithLifecycle()
    val timeRemaining by viewModel.timeRemaining.collectAsStateWithLifecycle()
    val liveAccel by viewModel.sensorCollector.liveAccel.collectAsStateWithLifecycle()
    val liveGyro by viewModel.sensorCollector.liveGyro.collectAsStateWithLifecycle()
    val exportPath by viewModel.exportPath.collectAsStateWithLifecycle()

    val currentBatch = batches.getOrNull(currentIndex)

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Data Collection", fontWeight = FontWeight.SemiBold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(24.dp)
                .fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            
            // Live Sensor Data Header
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "LIVE TELEMETRY ~ $participantId",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    
                    Text("Accelerometer (m/s²)", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("X: ${String.format("%.2f", liveAccel[0])}", color = Color.Gray)
                        Text("Y: ${String.format("%.2f", liveAccel[1])}", color = Color.Gray)
                        Text("Z: ${String.format("%.2f", liveAccel[2])}", color = Color.Gray)
                    }
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    HorizontalDivider()
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text("Gyroscope (rad/s)", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("X: ${String.format("%.2f", liveGyro[0])}", color = Color.Gray)
                        Text("Y: ${String.format("%.2f", liveGyro[1])}", color = Color.Gray)
                        Text("Z: ${String.format("%.2f", liveGyro[2])}", color = Color.Gray)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            when (recordingState) {
                RecordingState.Idle -> {
                    if (currentBatch != null) {
                        Text(
                            text = "Batch ${currentIndex + 1} of ${batches.size}",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = currentBatch.name.uppercase(),
                            style = MaterialTheme.typography.headlineLarge,
                            fontWeight = FontWeight.Black,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = "Duration: ${currentBatch.durationSec}s",
                            style = MaterialTheme.typography.titleMedium
                        )

                        Spacer(modifier = Modifier.weight(1f))

                        Button(
                            onClick = { viewModel.beginNextRecording() },
                            modifier = Modifier
                                .size(240.dp, 80.dp),
                            shape = RoundedCornerShape(40.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            Text("START BATCH", fontSize = 24.sp, fontWeight = FontWeight.Bold)
                        }
                        
                        Spacer(modifier = Modifier.height(24.dp))
                    }
                }
                RecordingState.Countdown -> {
                    Spacer(modifier = Modifier.weight(1f))
                    Text("GET READY", style = MaterialTheme.typography.headlineMedium, color = MaterialTheme.colorScheme.error, letterSpacing = 2.sp)
                    Text(
                        text = timeRemaining.toString(),
                        style = MaterialTheme.typography.displayLarge,
                        fontSize = 120.sp,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.error
                    )
                    Spacer(modifier = Modifier.weight(1f))
                }
                RecordingState.Recording -> {
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "RECORDING",
                        style = MaterialTheme.typography.headlineMedium,
                        color = Color(0xFF4CAF50),
                        letterSpacing = 2.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = currentBatch?.name?.uppercase() ?: "",
                        style = MaterialTheme.typography.titleLarge
                    )
                    Text(
                        text = timeRemaining.toString(),
                        style = MaterialTheme.typography.displayLarge,
                        fontSize = 100.sp,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    CircularProgressIndicator(
                        modifier = Modifier.size(64.dp),
                        color = Color(0xFF4CAF50),
                        strokeWidth = 6.dp
                    )
                    Spacer(modifier = Modifier.weight(1f))
                }
                RecordingState.Finished -> {
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "All Batches Completed!",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    if (exportPath == null) {
                        Button(
                            onClick = { viewModel.exportData() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(64.dp),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text("SAVE DATA TO CSV", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                        }
                    } else {
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("✅ Export Successful", fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32), fontSize = 18.sp)
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(exportPath ?: "", fontSize = 14.sp, textAlign = TextAlign.Center, color = Color.DarkGray)
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(32.dp))
                    Button(
                        onClick = { 
                            viewModel.resetSession()
                            navController.popBackStack("home", false) 
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.outlinedButtonColors(),
                        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.primary)
                    ) {
                        Text("RETURN TO HOME", color = MaterialTheme.colorScheme.primary)
                    }
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}