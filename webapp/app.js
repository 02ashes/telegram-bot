/* ========================================
   Inpaint Bot — Mini App Logic
   Canvas mask drawing + API integration
   ======================================== */

// ============================================================
// State
// ============================================================
let originalImage = null;        // HTMLImageElement
let currentTool = 'brush';       // 'brush' | 'eraser'
let brushSize = 20;
let isDrawing = false;
let canvasScale = 1;

// Canvas references
const mainCanvas = document.getElementById('mainCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const mainCtx = mainCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');

// UI references
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const fileInput = document.getElementById('fileInput');
const canvasSection = document.getElementById('canvasSection');
const promptSection = document.getElementById('promptSection');
const settingsSection = document.getElementById('settingsSection');
const generateSection = document.getElementById('generateSection');
const resultSection = document.getElementById('resultSection');

const brushBtn = document.getElementById('brushBtn');
const eraserBtn = document.getElementById('eraserBtn');
const clearBtn = document.getElementById('clearBtn');
const brushSizeSlider = document.getElementById('brushSize');
const brushSizeLabel = document.getElementById('brushSizeLabel');

const cfgSlider = document.getElementById('cfgSlider');
const cfgLabel = document.getElementById('cfgLabel');
const stepsSlider = document.getElementById('stepsSlider');
const stepsLabel = document.getElementById('stepsLabel');

const generateBtn = document.getElementById('generateBtn');
const progressInfo = document.getElementById('progressInfo');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');
const retryBtn = document.getElementById('retryBtn');

// ============================================================
// Telegram WebApp Init
// ============================================================
const tg = window.Telegram?.WebApp;
if (tg) {
    tg.ready();
    tg.expand();
    // Match Telegram theme if available
    document.body.style.backgroundColor = tg.themeParams?.bg_color || '#0f0f14';
}

// ============================================================
// Pod Status Check
// ============================================================
async function checkPodStatus() {
    try {
        const resp = await fetch('/api/pod-status', {
            headers: { 'ngrok-skip-browser-warning': '1' }
        });
        const data = await resp.json();
        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('.status-text');
        if (data.running) {
            dot.className = 'status-dot online';
            text.textContent = 'RunPod: Online';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'RunPod: Offline';
        }
    } catch (e) {
        // Server not reachable
    }
}

// Check every 15 seconds
checkPodStatus();
setInterval(checkPodStatus, 15000);

// ============================================================
// Image Upload
// ============================================================
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    loadImage(file);
});

// Drag-and-drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--accent)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    }
});

function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            originalImage = img;
            setupCanvases(img);

            // Show preview in upload area
            uploadPlaceholder.style.display = 'none';
            const preview = uploadArea.querySelector('img');
            if (preview) preview.remove();
            const previewImg = document.createElement('img');
            previewImg.src = e.target.result;
            uploadArea.appendChild(previewImg);

            // Show other sections
            canvasSection.style.display = '';
            promptSection.style.display = '';
            settingsSection.style.display = '';
            generateSection.style.display = '';
            resultSection.style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ============================================================
// Canvas Setup
// ============================================================
function setupCanvases(img) {
    const container = document.getElementById('canvasContainer');
    const containerWidth = container.clientWidth;

    // Scale image to fit container
    canvasScale = containerWidth / img.width;
    const displayW = containerWidth;
    const displayH = img.height * canvasScale;

    // Main canvas — shows the original image
    mainCanvas.width = img.width;
    mainCanvas.height = img.height;
    mainCanvas.style.width = displayW + 'px';
    mainCanvas.style.height = displayH + 'px';
    mainCtx.drawImage(img, 0, 0);

    // Mask canvas — transparent overlay for drawing mask
    maskCanvas.width = img.width;
    maskCanvas.height = img.height;
    maskCanvas.style.width = displayW + 'px';
    maskCanvas.style.height = displayH + 'px';
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

// ============================================================
// Mask Drawing
// ============================================================
function getCanvasPos(e) {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;

    let clientX, clientY;
    if (e.touches) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
    };
}

function drawAt(x, y) {
    const realBrushSize = brushSize / canvasScale;

    if (currentTool === 'brush') {
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.fillStyle = 'rgba(255, 50, 50, 0.45)';
        maskCtx.beginPath();
        maskCtx.arc(x, y, realBrushSize / 2, 0, Math.PI * 2);
        maskCtx.fill();
    } else {
        maskCtx.globalCompositeOperation = 'destination-out';
        maskCtx.beginPath();
        maskCtx.arc(x, y, realBrushSize / 2, 0, Math.PI * 2);
        maskCtx.fill();
        maskCtx.globalCompositeOperation = 'source-over';
    }
}

let lastPos = null;

function drawLine(from, to) {
    const realBrushSize = brushSize / canvasScale;
    const dist = Math.hypot(to.x - from.x, to.y - from.y);
    const steps = Math.max(1, Math.floor(dist / (realBrushSize / 4)));

    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        drawAt(x, y);
    }
}

// Mouse events
maskCanvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    isDrawing = true;
    const pos = getCanvasPos(e);
    lastPos = pos;
    drawAt(pos.x, pos.y);
});

maskCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getCanvasPos(e);
    if (lastPos) drawLine(lastPos, pos);
    lastPos = pos;
});

maskCanvas.addEventListener('mouseup', () => { isDrawing = false; lastPos = null; });
maskCanvas.addEventListener('mouseleave', () => { isDrawing = false; lastPos = null; });

// Touch events
maskCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const pos = getCanvasPos(e);
    lastPos = pos;
    drawAt(pos.x, pos.y);
}, { passive: false });

maskCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const pos = getCanvasPos(e);
    if (lastPos) drawLine(lastPos, pos);
    lastPos = pos;
}, { passive: false });

maskCanvas.addEventListener('touchend', () => { isDrawing = false; lastPos = null; });
maskCanvas.addEventListener('touchcancel', () => { isDrawing = false; lastPos = null; });

// ============================================================
// Tools
// ============================================================
brushBtn.addEventListener('click', () => {
    currentTool = 'brush';
    brushBtn.classList.add('active');
    eraserBtn.classList.remove('active');
});

eraserBtn.addEventListener('click', () => {
    currentTool = 'eraser';
    eraserBtn.classList.add('active');
    brushBtn.classList.remove('active');
});

clearBtn.addEventListener('click', () => {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
});

brushSizeSlider.addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value);
    brushSizeLabel.textContent = brushSize;
});

cfgSlider.addEventListener('input', (e) => {
    cfgLabel.textContent = parseFloat(e.target.value).toFixed(1);
});

stepsSlider.addEventListener('input', (e) => {
    stepsLabel.textContent = e.target.value;
});

// ============================================================
// Get mask as black/white image
// ============================================================
function getMaskDataURL() {
    // Create a temporary canvas for the mask
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = maskCanvas.width;
    tempCanvas.height = maskCanvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Get mask pixel data
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

    // Create black-and-white mask: white where painted, black elsewhere
    const output = tempCtx.createImageData(maskCanvas.width, maskCanvas.height);
    for (let i = 0; i < maskData.data.length; i += 4) {
        const alpha = maskData.data[i + 3]; // alpha channel
        const isWhite = alpha > 30;  // any painted area
        output.data[i] = isWhite ? 255 : 0;     // R
        output.data[i + 1] = isWhite ? 255 : 0; // G
        output.data[i + 2] = isWhite ? 255 : 0; // B
        output.data[i + 3] = 255;                // A
    }
    tempCtx.putImageData(output, 0, 0);

    return tempCanvas.toDataURL('image/png');
}

function getImageDataURL() {
    // Export original image from main canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = originalImage.width;
    tempCanvas.height = originalImage.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(originalImage, 0, 0);
    return tempCanvas.toDataURL('image/png');
}

// ============================================================
// Generate
// ============================================================
generateBtn.addEventListener('click', async () => {
    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) {
        alert('Введи промпт!');
        return;
    }

    if (!originalImage) {
        alert('Загрузи фото!');
        return;
    }

    // Check if mask has any content
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    let hasMask = false;
    for (let i = 3; i < maskData.data.length; i += 4) {
        if (maskData.data[i] > 30) { hasMask = true; break; }
    }
    if (!hasMask) {
        alert('Нарисуй маску!');
        return;
    }

    // Disable button, show progress
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    resultSection.style.display = 'none';

    // Progress animation
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 1, 90);
        progressFill.style.width = progress + '%';

        if (progress < 20) {
            progressText.textContent = '⚡ Запуск RunPod...';
            document.querySelector('.status-dot').className = 'status-dot starting';
            document.querySelector('.status-text').textContent = 'RunPod: Starting...';
        } else if (progress < 50) {
            progressText.textContent = '🧠 Загрузка модели...';
        } else {
            progressText.textContent = '🎨 Генерация...';
            document.querySelector('.status-dot').className = 'status-dot online';
            document.querySelector('.status-text').textContent = 'RunPod: Online';
        }
    }, 1000);

    try {
        // Get image and mask as base64
        const imageDataURL = getImageDataURL();
        const maskDataURL = getMaskDataURL();

        // Strip data:image/png;base64, prefix
        const imageB64 = imageDataURL.split(',')[1];
        const maskB64 = maskDataURL.split(',')[1];

        const cfg = parseFloat(cfgSlider.value);
        const steps = parseInt(stepsSlider.value);
        const negative = document.getElementById('negativeInput').value.trim();

        // Send request as JSON
        const payload = {
            image: imageB64,
            mask: maskB64,
            prompt: prompt,
            negative: negative,
            cfg: cfg,
            steps: steps,
        };

        const resp = await fetch('/api/inpaint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => null);
            throw new Error(errData?.error || errData?.detail || `Ошибка сервера: ${resp.status}`);
        }

        const data = await resp.json();

        clearInterval(progressInterval);

        if (data.error) {
            throw new Error(data.error);
        }

        // Show result
        progressFill.style.width = '100%';
        progressText.textContent = '✅ Готово!';

        resultImage.src = 'data:image/png;base64,' + data.image;
        resultSection.style.display = '';

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressText.textContent = '❌ Ошибка: ' + err.message;
        progressFill.style.width = '0%';
        alert('Ошибка: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// ============================================================
// Result Actions
// ============================================================
downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'inpaint_result.png';
    link.href = resultImage.src;
    link.click();
});

retryBtn.addEventListener('click', () => {
    resultSection.style.display = 'none';
    progressInfo.style.display = 'none';
    progressFill.style.width = '0%';
    // Scroll back to prompt
    promptSection.scrollIntoView({ behavior: 'smooth' });
});
