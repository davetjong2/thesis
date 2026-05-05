// --- KONFIGURASI ---
const GAS_URL = "https://script.google.com/macros/s/AKfycbz-PsaEOqX2bhGckXBbeL7ABsSzsoEQdNCZQdy6sUUSIP3DBvmqCXRf65BTbA48PaKDNw/exec";
const RECORDING_DURATION_MS = 20000; // 20 detik per skenario

const scenarios = [
    {
        id: "S1_Kanan_Samping",
        title: "1. TANGAN KANAN",
        img: "icons/01_tangan_kanan_samping_pinggang.png",
        desc: "Berjalan lurus secara natural dengan Handphone digenggam di tangan KANAN di samping pinggang."
    },
    {
        id: "S2_Kiri_Samping",
        title: "2. TANGAN KIRI",
        img: "icons/02_tangan_kiri_samping_pinggang.png.png",
        desc: "Berjalan lurus secara natural dengan Handphone digenggam di tangan KIRI di samping pinggang."
    },
    {
        id: "S3_Kanan_Main",
        title: "3. MAIN HP (KANAN)",
        img: "icons/03_tangan_kanan_main_hp.png",
        desc: "Berjalan lurus sambil fokus melihat dan mengetik layar Handphone dengan tangan KANAN saja."
    },
    {
        id: "S4_Kiri_Main",
        title: "4. MAIN HP (KIRI)",
        img: "icons/04_tangan_kiri_main_hp.png",
        desc: "Berjalan lurus sambil fokus melihat dan mengetik layar Handphone dengan tangan KIRI saja."
    },
    {
        id: "S5_DuaTangan_Main",
        title: "5. MAIN HP (KEDUA TANGAN)",
        img: "icons/05_kedua_tangan_main_hp.png",
        desc: "Berjalan lurus sambil fokus melihat dan mengetik layar Handphone menggunakan KEDUA TANGAN."
    }
];

// --- ELEMEN UI ---
const btnAction = document.getElementById('btn-action');
const statusText = document.getElementById('status-text');
const timerText = document.getElementById('timer-text');
const recordingStatus = document.getElementById('recording-status');
const sensorPreview = document.getElementById('sensor-preview');

const accelDataUI = document.getElementById('accel-data');
const gyroDataUI = document.getElementById('gyro-data');
const dataCountUI = document.getElementById('data-count');

const scenarioProgress = document.getElementById('scenario-progress');
const progressBar = document.getElementById('progress-bar');
const scenarioTitle = document.getElementById('scenario-title');
const scenarioImg = document.getElementById('scenario-img');
const scenarioDesc = document.getElementById('scenario-desc');

const imageModal = document.getElementById('image-modal');
const zoomedImg = document.getElementById('zoomed-img');
const zoomedDesc = document.getElementById('zoomed-desc');

// --- VARIABEL STATE ---
let isRecording = false;
let allSensorData = []; // Menggabungkan semua CSV row dari semua batch
let currentScenarioIndex = 0;
let startTime = 0;
let recordingInterval;
let autoStopTimeout;
let audioCtx = null;
let participantId = "";

let currentAccel = { x: 0, y: 0, z: 0 };
let currentGyro = { x: 0, y: 0, z: 0 };

// --- INISIALISASI ---
window.onload = () => {
    initParticipantId();
    initConsent();
    updateScenarioUI();
};

function initParticipantId() {
    let savedId = localStorage.getItem('gait_participant_id');
    if (!savedId) {
        const randomHex = Math.floor(Math.random() * 65536).toString(16).toUpperCase().padStart(4, '0');
        savedId = `P-${randomHex}`;
        localStorage.setItem('gait_participant_id', savedId);
    }
    participantId = savedId;
    document.getElementById('participant-id-display').innerText = participantId;
}

function initConsent() {
    const consentModal = document.getElementById('consent-modal');
    const consentCheckbox = document.getElementById('consent-checkbox');
    const btnAcceptConsent = document.getElementById('btn-accept-consent');

    const hasConsented = sessionStorage.getItem('gait_consented');
    if (hasConsented) {
        consentModal.classList.add('hidden');
    }

    consentCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            btnAcceptConsent.classList.remove('bg-slate-200', 'text-slate-400', 'cursor-not-allowed', 'pointer-events-none');
            btnAcceptConsent.classList.add('bg-indigo-600', 'text-white', 'shadow-[0_8px_20px_-6px_rgba(99,102,241,0.6)]', 'active:scale-95');
        } else {
            btnAcceptConsent.classList.add('bg-slate-200', 'text-slate-400', 'cursor-not-allowed', 'pointer-events-none');
            btnAcceptConsent.classList.remove('bg-indigo-600', 'text-white', 'shadow-[0_8px_20px_-6px_rgba(99,102,241,0.6)]', 'active:scale-95');
        }
    });

    btnAcceptConsent.addEventListener('click', () => {
        sessionStorage.setItem('gait_consented', 'true');
        consentModal.classList.add('opacity-0', 'pointer-events-none');
        setTimeout(() => consentModal.classList.add('hidden'), 300);
        
        if (!audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();
        }
    });
}

function updateScenarioUI() {
    if (currentScenarioIndex < scenarios.length) {
        const sc = scenarios[currentScenarioIndex];
        scenarioTitle.innerText = sc.title;
        scenarioImg.src = sc.img;
        scenarioDesc.innerText = sc.desc;
        
        const currentStep = currentScenarioIndex + 1;
        scenarioProgress.innerText = `${currentStep} / ${scenarios.length}`;
        progressBar.style.width = `${(currentStep / scenarios.length) * 100}%`;
        
        btnAction.innerText = `Mulai Skenario ${currentStep}`;
        btnAction.className = "w-full bg-gradient-to-r from-indigo-500 to-indigo-600 hover:from-indigo-600 hover:to-indigo-700 text-white font-bold py-4 px-6 rounded-2xl shadow-[0_8px_20px_-6px_rgba(99,102,241,0.6)] transition-all duration-200 active:scale-[0.97] active:shadow-none uppercase tracking-widest text-[13px]";
    } else {
        // Semua Skenario Selesai
        scenarioProgress.innerText = `5 / 5 (SELESAI)`;
        scenarioTitle.innerText = "SEMUA SKENARIO SELESAI";
        scenarioImg.src = "icons/01_tangan_kanan_samping_pinggang.png"; // fallback dummy
        scenarioImg.classList.add('opacity-50', 'grayscale');
        scenarioDesc.innerText = "Anda telah menyelesaikan kelima skenario perekaman data. Klik tombol di bawah ini untuk mengirimkan seluruh data ke database peneliti.";
        
        btnAction.innerText = `UNGGAH SELURUH DATA CSV`;
        btnAction.className = "w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-bold py-4 px-6 rounded-2xl shadow-[0_8px_20px_-6px_rgba(16,185,129,0.6)] transition-all duration-200 active:scale-[0.97] active:shadow-none uppercase tracking-widest text-[13px]";
    }
}

// --- AUDIO BEEP GENERATOR ---
function playBeep(frequency, durationMs, type = 'sine') {
    if (!audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContext();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    oscillator.type = type;
    oscillator.frequency.value = frequency;
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    oscillator.start();
    gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + durationMs / 1000);
    oscillator.stop(audioCtx.currentTime + durationMs / 1000);
}

// --- ZOOM MODAL LOGIC ---
function openImageModal() {
    if (currentScenarioIndex >= scenarios.length) return; // Disable zoom if done
    zoomedImg.src = scenarios[currentScenarioIndex].img;
    zoomedDesc.innerText = scenarios[currentScenarioIndex].desc;
    imageModal.classList.remove('opacity-0', 'pointer-events-none');
}

function closeImageModal() {
    imageModal.classList.add('opacity-0', 'pointer-events-none');
}

// --- LOGIKA UTAMA ---
btnAction.addEventListener('click', async () => {
    // Jika semua sudah selesai, maka tombol ini adalah tombol UPLOAD
    if (currentScenarioIndex >= scenarios.length) {
        await uploadCombinedData();
        return;
    }

    // Jika belum selesai, maka tombol ini adalah tombol MULAI SKENARIO
    const age = document.getElementById('age').value.trim();
    if (!age) {
        alert("Mohon isi Umur Anda terlebih dahulu sebelum memulai!");
        document.getElementById('age').focus();
        return;
    }

    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
            const permissionState = await DeviceMotionEvent.requestPermission();
            if (permissionState !== 'granted') {
                alert('Izin akses sensor ditolak. Tidak dapat merekam data.');
                return;
            }
        } catch (error) {
            console.error(error);
            alert('Gagal meminta izin sensor.');
            return;
        }
    }

    // Nonaktifkan form agar tidak diubah di tengah jalan
    document.getElementById('age').disabled = true;
    document.getElementById('gender').disabled = true;

    btnAction.classList.add('hidden');
    recordingStatus.classList.remove('hidden');
    sensorPreview.classList.remove('hidden');

    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }

    // COUNTDOWN
    let countdown = 3;
    statusText.innerText = "Bersiap...";
    statusText.className = "text-[11px] font-bold text-orange-500 uppercase tracking-widest";
    timerText.innerText = countdown;

    const countdownInterval = setInterval(() => {
        playBeep(440, 150); 
        countdown--;
        if (countdown > 0) {
            timerText.innerText = countdown;
        } else {
            clearInterval(countdownInterval);
            startRecordingCurrentScenario();
        }
    }, 1000);
});

function startRecordingCurrentScenario() {
    playBeep(880, 400); 
    
    isRecording = true;
    startTime = Date.now();
    
    statusText.innerText = `MEREKAM: ${scenarios[currentScenarioIndex].title}`;
    statusText.className = "text-[11px] font-bold text-rose-600 uppercase tracking-widest animate-pulse";
    
    window.addEventListener('devicemotion', handleMotion, true);

    recordingInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, RECORDING_DURATION_MS - elapsed);
        timerText.innerText = (remaining / 1000).toFixed(1);
    }, 100);

    autoStopTimeout = setTimeout(() => {
        finishCurrentScenario();
    }, RECORDING_DURATION_MS);
}

function handleMotion(event) {
    if (!isRecording) return;
    
    if (event.accelerationIncludingGravity) {
        currentAccel.x = event.accelerationIncludingGravity.x || 0;
        currentAccel.y = event.accelerationIncludingGravity.y || 0;
        currentAccel.z = event.accelerationIncludingGravity.z || 0;
    }
    
    if (event.rotationRate) {
        currentGyro.x = event.rotationRate.alpha || 0; 
        currentGyro.y = event.rotationRate.beta || 0;  
        currentGyro.z = event.rotationRate.gamma || 0; 
    }

    const timestamp = Date.now() - startTime;
    const currentScenarioId = scenarios[currentScenarioIndex].id;
    
    // Simpan ke array global dengan label skenario
    allSensorData.push({
        scenario: currentScenarioId,
        timestamp: timestamp,
        ax: currentAccel.x.toFixed(4),
        ay: currentAccel.y.toFixed(4),
        az: currentAccel.z.toFixed(4),
        gx: currentGyro.x.toFixed(4),
        gy: currentGyro.y.toFixed(4),
        gz: currentGyro.z.toFixed(4)
    });

    if (allSensorData.length % 5 === 0) {
        accelDataUI.innerText = `X: ${currentAccel.x.toFixed(2)} | Y: ${currentAccel.y.toFixed(2)} | Z: ${currentAccel.z.toFixed(2)}`;
        gyroDataUI.innerText = `X: ${currentGyro.x.toFixed(2)} | Y: ${currentGyro.y.toFixed(2)} | Z: ${currentGyro.z.toFixed(2)}`;
        dataCountUI.innerText = allSensorData.length;
    }
}

function finishCurrentScenario() {
    isRecording = false;
    window.removeEventListener('devicemotion', handleMotion, true);
    clearInterval(recordingInterval);
    clearTimeout(autoStopTimeout);
    
    timerText.innerText = "0.0";
    playBeep(300, 600, 'square'); 
    
    // Reset UI back to ready state
    recordingStatus.classList.add('hidden');
    sensorPreview.classList.add('hidden');
    btnAction.classList.remove('hidden');

    // Lanjut ke skenario berikutnya
    currentScenarioIndex++;
    updateScenarioUI();
}

async function uploadCombinedData() {
    if (allSensorData.length === 0) {
        alert("Tidak ada data sensor yang terkumpul!");
        return;
    }

    btnAction.innerText = "MENGUNGGAH...";
    btnAction.classList.remove('from-green-500', 'to-emerald-600');
    btnAction.classList.add('from-slate-400', 'to-slate-500', 'pointer-events-none');
    
    const age = document.getElementById('age').value.trim();
    const gender = document.getElementById('gender').value;
    
    // Gabungkan menjadi 1 CSV
    let csvString = "ParticipantId,Age,Gender,Scenario,TimestampMs,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ\n";
    allSensorData.forEach(row => {
        csvString += `${participantId},${age},${gender},${row.scenario},${row.timestamp},${row.ax},${row.ay},${row.az},${row.gx},${row.gy},${row.gz}\n`;
    });
    
    const filename = `GaitData_${participantId}_AllBatches_${new Date().getTime()}.csv`;
    
    try {
        const response = await fetch(GAS_URL, {
            method: 'POST',
            body: JSON.stringify({ filename, csvData: csvString })
        });
        
        const result = await response.json();
        
        if (result.status === "success") {
            btnAction.innerText = "BERHASIL DIUNGGAH!";
            btnAction.classList.remove('from-slate-400', 'to-slate-500');
            btnAction.classList.add('from-indigo-500', 'to-purple-600');
            setTimeout(() => {
                alert(`TERIMA KASIH!\nData ke-5 skenario berhasil digabung dan disimpan ke Google Drive Anda.\nTotal baris data: ${allSensorData.length}`);
                // Boleh direset jika mau dites ulang, atau biarkan selesai
            }, 500);
        } else {
            throw new Error(result.message);
        }
        
    } catch (error) {
        console.error("Upload error:", error);
        btnAction.innerText = "GAGAL MENGUNGGAH. KLIK UNTUK UNDUH MANUAL";
        btnAction.classList.remove('from-slate-400', 'to-slate-500', 'pointer-events-none');
        btnAction.classList.add('from-rose-500', 'to-red-600');
        
        btnAction.onclick = () => {
            downloadCSVManual(csvString, filename);
        };
    }
}

function downloadCSVManual(csvContent, filename) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
