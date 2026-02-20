/* ========================================
   Angel Arena — Mini App Logic
   Canvas mask drawing + Video generation
   ======================================== */

// ============================================================
// Auth — Telegram WebApp initData
// ============================================================
const tgInitData = window.Telegram?.WebApp?.initData || '';

// Expand webapp to full width
if (window.Telegram?.WebApp?.expand) {
    window.Telegram.WebApp.expand();
}

function authHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Telegram-Init-Data': tgInitData,
    };
}

function handleAuthError(resp) {
    if (resp.status === 401) {
        alert('❌ Access denied. Enter invite code in bot: /invite CODE');
        return true;
    }
    return false;
}

// ============================================================
// State
// ============================================================
let currentMode = 'inpaint'; // 'inpaint', 'video', 'image', or 'dark'
let darkQuality = 'fast'; // 'fast' or 'detailed'
let darkMode = 'edit'; // 'edit' or 'generate'
let batchCount = 1; // 1-4
let currentTool = 'brush';
let brushSize = 20;
let isDrawing = false;
let originalImage = null;
let originalImage2 = null;  // reference image for Image mode
let mainCtx = null;
let maskCtx = null;
let galleryItems = []; // [{dataUrl, timestamp}]

// ============================================================
// DOM Elements
// ============================================================
const mainCanvas = document.getElementById('mainCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const fileInput = document.getElementById('fileInput');

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
const resultSection = document.getElementById('resultSection');
const resultImage = document.getElementById('resultImage');
const resultVideo = document.getElementById('resultVideo');
const downloadBtn = document.getElementById('downloadBtn');
const retryBtn = document.getElementById('retryBtn');

// Video settings
const framesSlider = document.getElementById('framesSlider');
const framesLabel = document.getElementById('framesLabel');
const fpsSelect = document.getElementById('fpsSelect');
const resolutionSelect = document.getElementById('resolutionSelect');
const audioToggle = document.getElementById('audioToggle');
const audioSettings = document.getElementById('audioSettings');

// Image edit settings
const denoiseSlider = document.getElementById('denoiseSlider');
const denoiseLabel = document.getElementById('denoiseLabel');
const editStepsSlider = document.getElementById('editStepsSlider');
const editStepsLabel = document.getElementById('editStepsLabel');
const uploadArea2 = document.getElementById('uploadArea2');
const uploadPlaceholder2 = document.getElementById('uploadPlaceholder2');
const fileInput2 = document.getElementById('fileInput2');

// Dark Beast settings
const darkDenoiseSlider = document.getElementById('darkDenoiseSlider');
const darkDenoiseLabel = document.getElementById('darkDenoiseLabel');
const darkStepsSlider = document.getElementById('darkStepsSlider');
const darkStepsLabel = document.getElementById('darkStepsLabel');

// Tabs
const tabInpaint = document.getElementById('tabInpaint');
const tabVideo = document.getElementById('tabVideo');
const tabImage = document.getElementById('tabImage');
const tabDark = document.getElementById('tabDark');

// ============================================================
// Telegram WebApp
// ============================================================
const tg = window.Telegram?.WebApp;
if (tg) {
    tg.ready();
    tg.expand();
    document.body.style.backgroundColor = tg.themeParams?.bg_color || '#0f0f14';
}

// ============================================================
// Serverless Status
// ============================================================
(function setServerlessStatus() {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (dot) dot.className = 'status-dot online';
    if (text) text.textContent = '⚡ Serverless';
})();

// ============================================================
// Mode Tabs
// ============================================================
tabInpaint.addEventListener('click', () => switchMode('inpaint'));
tabVideo.addEventListener('click', () => switchMode('video'));
tabImage.addEventListener('click', () => switchMode('image'));
tabDark.addEventListener('click', () => switchMode('dark'));

function switchMode(mode) {
    currentMode = mode;

    // Update tabs
    tabInpaint.classList.toggle('active', mode === 'inpaint');
    tabVideo.classList.toggle('active', mode === 'video');
    tabImage.classList.toggle('active', mode === 'image');
    tabDark.classList.toggle('active', mode === 'dark');

    // Show/hide mode-specific sections
    document.querySelectorAll('.inpaint-only').forEach(el => {
        el.style.display = mode === 'inpaint' && originalImage ? '' : 'none';
    });
    document.querySelectorAll('.video-only').forEach(el => {
        el.style.display = mode === 'video' && originalImage ? '' : 'none';
    });
    document.querySelectorAll('.image-only').forEach(el => {
        el.style.display = mode === 'image' && originalImage ? '' : 'none';
    });
    document.querySelectorAll('.dark-only').forEach(el => {
        el.style.display = mode === 'dark' && originalImage ? '' : 'none';
    });

    // Update prompt placeholder
    const promptInput = document.getElementById('promptInput');
    if (mode === 'inpaint') {
        promptInput.placeholder = 'Describe what to paint in the mask...';
    } else if (mode === 'video') {
        promptInput.placeholder = 'Describe motion (woman slowly turns her head, smiles...)';
    } else if (mode === 'dark') {
        promptInput.placeholder = 'Dark Beast: describe NSFW edit...';
    } else {
        promptInput.placeholder = 'Describe the edit (add cum on face, finger in ass, remove clothes...)';
    }

    // Update negative prompt default
    const negativeInput = document.getElementById('negativeInput');
    if (mode === 'video' && negativeInput.value === 'blurry, ugly, deformed, watermark, text, low quality, cartoon') {
        negativeInput.value = '';
    } else if ((mode === 'inpaint' || mode === 'image' || mode === 'dark') && negativeInput.value === '') {
        negativeInput.value = 'blurry, ugly, deformed, watermark, text, low quality, cartoon';
    }

    // Update generate button text
    const btnText = generateBtn.querySelector('.btn-text');
    if (mode === 'inpaint') btnText.textContent = '🚀 Generate';
    else if (mode === 'video') btnText.textContent = '🎬 Generate Video';
    else if (mode === 'dark') btnText.textContent = '🖤 Dark Beast';
    else btnText.textContent = '🖼️ Edit Image';

    // Hide result if mode changed
    resultSection.style.display = 'none';

    // Update preset buttons
    renderPresets();
}

// ============================================================
// Upload
// ============================================================
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadImage(file);
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(168, 85, 247, 0.6)';
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
        const img = new window.Image();
        img.onload = () => {
            originalImage = img;

            // Show preview in upload area
            uploadPlaceholder.style.display = 'none';
            let previewImg = uploadArea.querySelector('img');
            if (!previewImg) {
                previewImg = document.createElement('img');
                uploadArea.appendChild(previewImg);
            }
            previewImg.src = e.target.result;

            // Setup canvases for inpaint mode
            setupCanvases(img);

            // Show relevant sections
            document.getElementById('canvasSection').style.display = currentMode === 'inpaint' ? '' : 'none';
            document.getElementById('promptSection').style.display = '';
            document.getElementById('settingsSection').style.display = currentMode === 'inpaint' ? '' : 'none';
            document.getElementById('generateSection').style.display = '';

            // Show video-only sections if in video mode
            document.querySelectorAll('.video-only').forEach(el => {
                el.style.display = currentMode === 'video' ? '' : 'none';
            });
            document.querySelectorAll('.inpaint-only').forEach(el => {
                el.style.display = currentMode === 'inpaint' ? '' : 'none';
            });
            document.querySelectorAll('.image-only').forEach(el => {
                el.style.display = currentMode === 'image' ? '' : 'none';
            });
            document.querySelectorAll('.dark-only').forEach(el => {
                el.style.display = currentMode === 'dark' ? '' : 'none';
            });
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
    const maxW = container.clientWidth || 460;
    const scale = Math.min(maxW / img.width, 1);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);

    mainCanvas.width = w;
    mainCanvas.height = h;
    maskCanvas.width = w;
    maskCanvas.height = h;

    mainCtx = mainCanvas.getContext('2d');
    maskCtx = maskCanvas.getContext('2d');

    mainCtx.drawImage(img, 0, 0, w, h);
    maskCtx.clearRect(0, 0, w, h);
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
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);

    if (currentTool === 'brush') {
        maskCtx.fillStyle = 'rgba(255, 0, 100, 0.45)';
        maskCtx.fill();
    } else {
        maskCtx.save();
        maskCtx.globalCompositeOperation = 'destination-out';
        maskCtx.fillStyle = 'rgba(0,0,0,1)';
        maskCtx.fill();
        maskCtx.restore();
    }
}

let lastPos = null;

function drawLine(from, to) {
    const dist = Math.sqrt((to.x - from.x) ** 2 + (to.y - from.y) ** 2);
    const steps = Math.max(Math.ceil(dist / (brushSize / 4)), 1);
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
    drawLine(lastPos, pos);
    lastPos = pos;
});

window.addEventListener('mouseup', () => {
    isDrawing = false;
    lastPos = null;
});

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
    drawLine(lastPos, pos);
    lastPos = pos;
}, { passive: false });

window.addEventListener('touchend', () => {
    isDrawing = false;
    lastPos = null;
});

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
    if (maskCtx) {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
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

framesSlider.addEventListener('input', (e) => {
    framesLabel.textContent = e.target.value;
});

// Audio toggle
audioToggle.addEventListener('change', () => {
    audioSettings.style.display = audioToggle.checked ? '' : 'none';
});

// Denoise slider
denoiseSlider.addEventListener('input', (e) => {
    denoiseLabel.textContent = parseFloat(e.target.value).toFixed(2);
});

editStepsSlider.addEventListener('input', (e) => {
    editStepsLabel.textContent = e.target.value;
});

// LoRA strength slider
const loraStrengthSlider = document.getElementById('loraStrengthSlider');
const loraStrengthLabel = document.getElementById('loraStrengthLabel');
if (loraStrengthSlider) {
    loraStrengthSlider.addEventListener('input', (e) => {
        loraStrengthLabel.textContent = parseFloat(e.target.value).toFixed(1);
    });
}

// Dark Beast sliders
if (darkDenoiseSlider) {
    darkDenoiseSlider.addEventListener('input', (e) => {
        darkDenoiseLabel.textContent = parseFloat(e.target.value).toFixed(2);
    });
}
if (darkStepsSlider) {
    darkStepsSlider.addEventListener('input', (e) => {
        darkStepsLabel.textContent = e.target.value;
    });
}

// Quality toggle (Fast / Detailed)
document.querySelectorAll('#darkQualitySection .quality-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#darkQualitySection .quality-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        darkQuality = btn.dataset.quality;
    });
});

// Dark mode toggle (Edit / Generate)
document.querySelectorAll('#darkModeSection .quality-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#darkModeSection .quality-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        darkMode = btn.dataset.darkmode;
    });
});

// ============================================================
// Upload 2 (Reference Image for Image mode) — optional, elements may not exist
// ============================================================
if (uploadArea2 && fileInput2) {
    uploadArea2.addEventListener('click', () => fileInput2.click());

    fileInput2.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            loadImage2(file);
        }
    });

    uploadArea2.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea2.style.borderColor = '#a855f7';
    });

    uploadArea2.addEventListener('dragleave', () => {
        uploadArea2.style.borderColor = '';
    });

    uploadArea2.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea2.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadImage2(file);
        }
    });
}

function loadImage2(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new window.Image();
        img.onload = () => {
            originalImage2 = img;
            uploadPlaceholder2.innerHTML = `<img src="${e.target.result}" style="max-width:100%;max-height:150px;border-radius:8px;">`;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function getImage2DataURL() {
    if (!originalImage2) return null;
    const c = document.createElement('canvas');
    c.width = originalImage2.naturalWidth;
    c.height = originalImage2.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(originalImage2, 0, 0);
    return c.toDataURL('image/png');
}

// ============================================================
// Get mask as black/white image
// ============================================================
function getMaskDataURL() {
    const w = maskCanvas.width;
    const h = maskCanvas.height;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill black (keep)
    tempCtx.fillStyle = '#000';
    tempCtx.fillRect(0, 0, w, h);

    // Get mask pixels
    const maskData = maskCtx.getImageData(0, 0, w, h);
    const tempData = tempCtx.getImageData(0, 0, w, h);

    for (let i = 0; i < maskData.data.length; i += 4) {
        if (maskData.data[i + 3] > 10) {
            tempData.data[i] = 255;
            tempData.data[i + 1] = 255;
            tempData.data[i + 2] = 255;
            tempData.data[i + 3] = 255;
        }
    }
    tempCtx.putImageData(tempData, 0, 0);
    return tempCanvas.toDataURL('image/png');
}

function getImageDataURL() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = originalImage.width;
    tempCanvas.height = originalImage.height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(originalImage, 0, 0);
    return tempCanvas.toDataURL('image/png');
}

// ============================================================
// Generate
// ============================================================
generateBtn.addEventListener('click', async () => {
    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) {
        alert('Enter a prompt!');
        return;
    }

    if (!originalImage) {
        alert('Upload a photo first!');
        return;
    }

    // Video mode: always 1 (too slow for batch)
    const count = (currentMode === 'video') ? 1 : batchCount;

    for (let i = 0; i < count; i++) {
        if (count > 1) {
            progressInfo.style.display = '';
            progressText.textContent = `🔄 Generating ${i + 1}/${count}...`;
        }

        if (currentMode === 'inpaint') {
            await generateInpaint(prompt);
        } else if (currentMode === 'video') {
            await generateVideo(prompt);
        } else if (currentMode === 'dark') {
            await generateDarkEdit(prompt);
        } else {
            await generateImageEdit(prompt);
        }
    }
});

// ============================================================
// Inpaint Generation
// ============================================================
async function generateInpaint(prompt) {
    const negative = document.getElementById('negativeInput').value;
    const cfg = parseFloat(cfgSlider.value);
    const steps = parseInt(stepsSlider.value);

    // UI state
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    resultSection.style.display = 'none';

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 1, 90);
        progressFill.style.width = progress + '%';

        if (progress < 20) {
            progressText.textContent = '⚡ Starting...';
        } else if (progress < 50) {
            progressText.textContent = '🧠 Preparing...';
        } else {
            progressText.textContent = '🎨 Generating...';
        }
    }, 1000);

    try {
        const imageDataURL = getImageDataURL();
        const maskDataURL = getMaskDataURL();

        const imageB64 = imageDataURL.split(',')[1];
        const maskB64 = maskDataURL.split(',')[1];

        const resp = await fetch('/api/inpaint', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({
                image: imageB64,
                mask: maskB64,
                prompt: prompt,
                negative: negative,
                cfg: cfg,
                steps: steps,
            }),
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => null);
            throw new Error(errData?.error || errData?.detail || `Server error: ${resp.status}`);
        }

        const data = await resp.json();

        clearInterval(progressInterval);

        if (data.error) {
            throw new Error(data.error);
        }

        progressFill.style.width = '100%';
        progressText.textContent = '✅ Done!';

        resultImage.src = 'data:image/png;base64,' + data.image;
        addToGallery(resultImage.src);
        resultImage.style.display = '';
        resultVideo.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        progressText.textContent = '❌ ' + err.message;
        alert('Error: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ============================================================
// Video Generation
// ============================================================
async function generateVideo(prompt) {
    const negative = document.getElementById('negativeInput').value;
    const frames = parseInt(framesSlider.value);
    const fps = parseInt(fpsSelect.value);
    const resVal = resolutionSelect.value;
    let width = 0, height = 0;
    if (resVal !== 'auto') {
        const resolution = resVal.split('x');
        width = parseInt(resolution[0]);
        height = parseInt(resolution[1]);
    }
    const audioEnabled = audioToggle.checked;
    const audioPrompt = document.getElementById('audioPromptInput')?.value || '';
    const audioNegative = document.getElementById('audioNegativeInput')?.value || 'music, speech, talking, noise, static';

    // UI state
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    resultSection.style.display = 'none';

    const videoDuration = frames / fps;
    const estimatedTime = Math.max(60, frames * 3); // rough estimate

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + (90 / (estimatedTime)), 90);
        progressFill.style.width = progress + '%';

        if (progress < 10) {
            progressText.textContent = '⚡ Starting...';
        } else if (progress < 25) {
            progressText.textContent = '🧠 Preparing...';
        } else if (progress < 80) {
            progressText.textContent = `🎬 Generating video (${frames} frames, ~${videoDuration.toFixed(1)}s)...`;
        } else if (audioEnabled) {
            progressText.textContent = '🔊 Generating audio...';
        } else {
            progressText.textContent = '📦 Encoding mp4...';
        }
    }, 1000);

    try {
        const imageDataURL = getImageDataURL();
        const imageB64 = imageDataURL.split(',')[1];

        const body = {
            image: imageB64,
            prompt: prompt,
            negative: negative,
            frames: frames,
            fps: fps,
            width: width,
            height: height,
            audio_enabled: audioEnabled,
        };

        if (audioEnabled) {
            body.audio_prompt = audioPrompt;
            body.audio_negative = audioNegative;
        }

        const resp = await fetch('/api/video', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });

        console.log('Video response status:', resp.status);

        if (!resp.ok) {
            const errData = await resp.json().catch(() => null);
            throw new Error(errData?.error || errData?.detail || `Server error: ${resp.status}`);
        }

        console.log('Parsing video response JSON...');
        const text = await resp.text();
        console.log('Response text length:', text.length);
        const data = JSON.parse(text);
        console.log('Video response keys:', Object.keys(data));
        console.log('Has video key:', !!data.video);
        console.log('Video base64 length:', data.video?.length || 0);

        clearInterval(progressInterval);

        if (data.error) {
            throw new Error(data.error);
        }

        progressFill.style.width = '100%';
        progressText.textContent = '✅ Video ready!';

        // Show video result
        const videoBlob = base64ToBlob(data.video, 'video/mp4');
        console.log('Video blob size:', videoBlob.size);
        const videoUrl = URL.createObjectURL(videoBlob);
        resultVideo.src = videoUrl;
        resultVideo.style.display = '';
        resultImage.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        progressText.textContent = '❌ ' + err.message;
        alert('Error: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ============================================================
// Image Edit Generation (Flux 2 Klein)
// ============================================================
async function generateImageEdit(prompt) {
    const negative = document.getElementById('negativeInput').value.trim();
    const denoise = parseFloat(denoiseSlider.value);
    const steps = parseInt(editStepsSlider.value);
    const imageDataURL = getImageDataURL();
    const image2DataURL = getImage2DataURL();

    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    progressFill.style.width = '10%';
    progressText.textContent = '🚀 Sending...';
    resultSection.style.display = 'none';

    let progressInterval;

    try {
        progressInterval = setInterval(() => {
            const cur = parseFloat(progressFill.style.width);
            if (cur < 85) {
                progressFill.style.width = (cur + 1.5) + '%';
                if (cur > 30) progressText.textContent = '⏳ Generating...';
                if (cur > 70) progressText.textContent = '🔄 Almost done...';
            }
        }, 1500);

        const body = {
            image: imageDataURL.split(',')[1],
            prompt: prompt,
            negative: negative,
            denoise: denoise,
            steps: steps,
        };

        // LoRA (optional)
        const loraName = document.getElementById('loraNameInput')?.value?.trim() || '';
        const loraStrength = parseFloat(document.getElementById('loraStrengthSlider')?.value || 1.0);
        if (loraName) {
            body.lora_name = loraName;
            body.lora_strength = loraStrength;
        }

        if (image2DataURL) {
            body.image2 = image2DataURL.split(',')[1];
        }

        const response = await fetch('/api/image-edit', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });

        clearInterval(progressInterval);

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        progressFill.style.width = '100%';
        progressText.textContent = '✅ Done!';

        // Show image result
        resultImage.src = 'data:image/png;base64,' + data.image;
        addToGallery(resultImage.src);
        resultImage.style.display = '';
        resultVideo.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        progressText.textContent = '❌ ' + err.message;
        alert('Error: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ============================================================
// Dark Beast Generation
// ============================================================
async function generateDarkEdit(prompt) {
    const negative = document.getElementById('negativeInput').value.trim();
    const denoise = parseFloat(darkDenoiseSlider.value);
    const steps = parseInt(darkStepsSlider.value);
    const imageDataURL = getImageDataURL();

    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    progressFill.style.width = '10%';
    progressText.textContent = '🖤 Dark Beast...';
    resultSection.style.display = 'none';

    let progressInterval;

    try {
        progressInterval = setInterval(() => {
            const cur = parseFloat(progressFill.style.width);
            if (cur < 85) {
                progressFill.style.width = (cur + 1.5) + '%';
                if (cur > 30) progressText.textContent = '✨ Generating...';
                if (cur > 70) progressText.textContent = '🔥 Almost done...';
            }
        }, 1500);

        const body = {
            image: imageDataURL.split(',')[1],
            prompt: prompt,
            negative: negative,
            denoise: denoise,
            steps: steps,
            quality: darkQuality,
            mode: darkMode,
        };

        const response = await fetch('/api/image-edit-dark', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });

        clearInterval(progressInterval);

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        progressFill.style.width = '100%';
        progressText.textContent = '✅ Done!';

        resultImage.src = 'data:image/png;base64,' + data.image;
        addToGallery(resultImage.src);
        resultImage.style.display = '';
        resultVideo.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        progressText.textContent = '❌ ' + err.message;
        alert('Error: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ============================================================
// Utility: base64 to Blob
// ============================================================
function base64ToBlob(b64, type) {
    const byteChars = atob(b64);
    const byteArrays = [];
    for (let offset = 0; offset < byteChars.length; offset += 512) {
        const slice = byteChars.slice(offset, offset + 512);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: type });
}

// ============================================================
// Download, Copy & Retry
// ============================================================
downloadBtn.addEventListener('click', () => {
    if ((currentMode === 'inpaint' || currentMode === 'image' || currentMode === 'dark') && resultImage.src) {
        const a = document.createElement('a');
        a.href = resultImage.src;
        const names = { inpaint: 'inpaint_result.png', image: 'edit_result.png', dark: 'dark_result.png' };
        a.download = names[currentMode] || 'result.png';
        a.click();
    } else if (currentMode === 'video' && resultVideo.src) {
        const a = document.createElement('a');
        a.href = resultVideo.src;
        a.download = 'video_result.mp4';
        a.click();
    }
});

async function copyImageToClipboard(imgSrc) {
    try {
        const resp = await fetch(imgSrc);
        const blob = await resp.blob();
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
        return true;
    } catch (e) {
        console.error('Copy failed:', e);
        return false;
    }
}

const copyBtn = document.getElementById('copyBtn');
if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
        if (resultImage.src) {
            const ok = await copyImageToClipboard(resultImage.src);
            copyBtn.textContent = ok ? '✅ Copied!' : '❌ Failed';
            setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 1500);
        }
    });
}

retryBtn.addEventListener('click', () => {
    resultSection.style.display = 'none';
    progressFill.style.width = '0%';
    progressInfo.style.display = 'none';
    document.getElementById('generateSection').scrollIntoView({ behavior: 'smooth' });
});

// ============================================================
// Batch Count Selector
// ============================================================
document.querySelectorAll('.batch-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.batch-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        batchCount = parseInt(btn.dataset.count);
    });
});

// ============================================================
// Ctrl+V Paste Support
// ============================================================
document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            if (file) loadImage(file);
            break;
        }
    }
});

// ============================================================
// Gallery
// ============================================================
const gallerySection = document.getElementById('gallerySection');
const galleryStrip = document.getElementById('galleryStrip');
const lightbox = document.getElementById('galleryLightbox');
const lightboxImage = document.getElementById('lightboxImage');
const lightboxOverlay = document.getElementById('lightboxOverlay');
const lightboxClose = document.getElementById('lightboxClose');
const lightboxDownload = document.getElementById('lightboxDownload');
const lightboxCopy = document.getElementById('lightboxCopy');
const lightboxDelete = document.getElementById('lightboxDelete');
let activeLightboxIndex = -1;

function addToGallery(dataUrl) {
    galleryItems.push({ dataUrl, timestamp: Date.now() });
    renderGallery();
}

function renderGallery() {
    if (!galleryStrip) return;
    if (galleryItems.length === 0) {
        if (gallerySection) gallerySection.style.display = 'none';
        return;
    }
    if (gallerySection) gallerySection.style.display = '';
    galleryStrip.innerHTML = '';
    galleryItems.forEach((item, idx) => {
        const thumb = document.createElement('img');
        thumb.className = 'gallery-thumb';
        thumb.src = item.dataUrl;
        thumb.addEventListener('click', () => openLightbox(idx));
        galleryStrip.appendChild(thumb);
    });
    // Scroll to end
    galleryStrip.scrollLeft = galleryStrip.scrollWidth;
}

function openLightbox(idx) {
    activeLightboxIndex = idx;
    lightboxImage.src = galleryItems[idx].dataUrl;
    lightbox.style.display = '';
}

function closeLightbox() {
    lightbox.style.display = 'none';
    activeLightboxIndex = -1;
}

if (lightboxOverlay) lightboxOverlay.addEventListener('click', closeLightbox);
if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);

if (lightboxDownload) {
    lightboxDownload.addEventListener('click', () => {
        if (activeLightboxIndex < 0) return;
        const a = document.createElement('a');
        a.href = galleryItems[activeLightboxIndex].dataUrl;
        a.download = `gallery_${activeLightboxIndex + 1}.png`;
        a.click();
    });
}

if (lightboxCopy) {
    lightboxCopy.addEventListener('click', async () => {
        if (activeLightboxIndex < 0) return;
        const ok = await copyImageToClipboard(galleryItems[activeLightboxIndex].dataUrl);
        lightboxCopy.textContent = ok ? '✅ Copied!' : '❌ Failed';
        setTimeout(() => { lightboxCopy.textContent = '📋 Copy'; }, 1500);
    });
}

if (lightboxDelete) {
    lightboxDelete.addEventListener('click', () => {
        if (activeLightboxIndex < 0) return;
        galleryItems.splice(activeLightboxIndex, 1);
        closeLightbox();
        renderGallery();
    });
}

// ============================================================
// Preset Prompt Buttons
// ============================================================
const PRESETS = {
    inpaint: [
        {
            label: '💋 Name',
            prompt: 'The word "Name" handwritten in dark crimson lipstick on bare skin. Semi-transparent smeared lipstick, skin texture visible through the letters. Faded messy imperfect handwritten letters, slightly uneven and crooked. Dark burgundy red lipstick lightly applied on skin. Photorealistic.',
            negative: 'blurry, ugly, deformed, font, low quality, cartoon',
        },
    ],
    dark: [
        {
            label: '💦 Cum',
            prompt: 'thick white cum dripping on skin, semen splattered, creampie leaking, wet glistening cum drops, realistic bodily fluid, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '💛 Pee',
            prompt: 'clear transparent stream of pee flowing between legs, warm liquid dripping on thighs, wet glistening skin, watersports, clear fluid like water, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '💩 Shit',
            prompt: 'brown feces smeared on skin, dirty messy scat, soiled body, realistic texture, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '🔞 Nude',
            prompt: 'completely naked, fully nude, no clothes, bare breasts with erect nipples, exposed pussy, smooth skin, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, clothes, dressed, fabric',
        },
        {
            label: '🍆 Anal',
            prompt: 'man\'s thick cock deep inside her ass, anal penetration from behind, stretched anus around penis, doggy style anal sex, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, extra limbs',
        },
    ],
    image: [
        {
            label: '\ud83d\udca6 Cum',
            prompt: 'thick white cum dripping on skin, semen splattered, wet glistening cum drops, realistic bodily fluid, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '\ud83d\udc9b Pee',
            prompt: 'clear transparent stream of pee flowing between legs, warm liquid dripping on thighs, wet glistening skin, watersports, clear fluid like water, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '\ud83d\udca9 Shit',
            prompt: 'brown feces smeared on skin, dirty messy scat, soiled body, realistic texture, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: '\ud83d\udd1e Nude',
            prompt: 'completely naked, fully nude, no clothes, bare breasts with erect nipples, exposed pussy, smooth skin, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, clothes, dressed, fabric',
        },
        {
            label: '\ud83c\udf46 Anal',
            prompt: 'man\'s thick cock deep inside her ass, anal penetration from behind, stretched anus around penis, doggy style anal sex, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, extra limbs',
        },
    ],
    video: [],
};

function renderPresets() {
    const row = document.getElementById('presetRow');
    if (!row) return;
    const presets = PRESETS[currentMode] || [];
    row.innerHTML = '';
    presets.forEach(p => {
        const btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.textContent = p.label;
        btn.addEventListener('click', () => {
            document.getElementById('promptInput').value = p.prompt;
            document.getElementById('negativeInput').value = p.negative;
        });
        row.appendChild(btn);
    });
}

// Initial render
renderPresets();
