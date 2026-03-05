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
let editSubmode = 'default'; // 'default' (no depth/canny) or 'depth' (preserves shape)
let darkQuality = 'fast'; // 'fast' or 'detailed'
let darkMode = 'edit'; // 'edit' or 'generate'
let darkResolution = '768x1344'; // resolution for Generate mode (9:16)
let darkGenSubmode = 'default'; // 'default' or 'faceswap'
let faceImageB64 = null; // base64 face photo for BFS
let batchCount = 1; // 1-4
let currentTool = 'brush';
let brushSize = 20;
let isDrawing = false;
let originalImage = null;
let originalImage2 = null;  // reference image for Image mode
let darkImage2 = null;      // second image for Dark mode (combine two girls)
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

// Dark Image 2 elements
const darkImg2Add = document.getElementById('darkImg2Add');
const darkImg2Preview = document.getElementById('darkImg2Preview');
const darkImg2Img = document.getElementById('darkImg2Img');
const darkImg2Remove = document.getElementById('darkImg2Remove');
const darkFileInput2 = document.getElementById('darkFileInput2');

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
    // Force pure black — ignore Telegram's dark-blue theme
    document.body.style.backgroundColor = '#000000';
    try { tg.setBackgroundColor('#000000'); } catch (_) { }
    try { tg.setHeaderColor('#000000'); } catch (_) { }
}

// ============================================================
// Serverless Status
// ============================================================
(function setServerlessStatus() {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (dot) dot.className = 'status-dot online';
    if (text) text.textContent = 'Serverless';
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

    // Dark Generate can work without image (text2img)
    const darkGenNoImage = (mode === 'dark' && darkMode === 'generate');

    // Bug #2: Hide upload section in Dark Generate mode (text2img doesn't need photo)
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.style.display = darkGenNoImage ? 'none' : '';
    }

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

    // Edit sub-mode section: show when image tab is active
    if (mode === 'image') {
        let editSubmodeSection = document.getElementById('editSubmodeSection');
        if (!editSubmodeSection && originalImage) {
            // Dynamically create the sub-mode toggle (since HTML is deployed)
            editSubmodeSection = document.createElement('div');
            editSubmodeSection.id = 'editSubmodeSection';
            editSubmodeSection.className = 'section';
            editSubmodeSection.innerHTML = `
                <div class="section-title sub">Edit Mode</div>
                <div class="quality-toggle" id="editSubmodeToggle">
                    <button class="quality-btn ${editSubmode === 'default' ? 'active' : ''}" data-submode="default">⚡ Default</button>
                    <button class="quality-btn ${editSubmode === 'depth' ? 'active' : ''}" data-submode="depth">🎯 Depth</button>
                </div>
            `;
            // Insert after prompt section
            const promptSection = document.getElementById('promptSection');
            if (promptSection) promptSection.after(editSubmodeSection);
            // Bind toggle buttons
            editSubmodeSection.querySelectorAll('.quality-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    editSubmodeSection.querySelectorAll('.quality-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    editSubmode = btn.dataset.submode;
                    switchMode('image');
                });
            });
        }
        if (editSubmodeSection) editSubmodeSection.style.display = originalImage ? '' : 'none';
    } else {
        const editSubmodeSection = document.getElementById('editSubmodeSection');
        if (editSubmodeSection) editSubmodeSection.style.display = 'none';
    }
    document.querySelectorAll('.dark-only').forEach(el => {
        el.style.display = mode === 'dark' && (originalImage || darkGenNoImage) ? '' : 'none';
    });

    // AFTER dark-only loop: override darkImage2Section for Edit mode too
    if (mode === 'image') {
        // Bug #2: Only show image2 slot in Default mode (Depth doesn't need it)
        const img2Section = document.getElementById('darkImage2Section');
        if (img2Section) img2Section.style.display = (editSubmode === 'default' && originalImage) ? '' : 'none';

        // Show/hide image edit settings (denoise+steps) based on submode
        const imageSettings = document.getElementById('imageSettingsSection');
        if (imageSettings) imageSettings.style.display = (editSubmode === 'depth' && originalImage) ? '' : 'none';
    }

    // Dark Mode/Quality toggles always show when Dark tab is active
    if (mode === 'dark') {
        const modeSection = document.getElementById('darkModeSection');
        if (modeSection) modeSection.style.display = '';
        // Quality only for Edit mode
        const qualitySection = document.getElementById('darkQualitySection');
        if (qualitySection) qualitySection.style.display = darkMode === 'edit' ? '' : 'none';
        // Hide Reference image section in Generate mode (text2img, no i2i)
        const img2Section = document.getElementById('darkImage2Section');
        if (img2Section) img2Section.style.display = (darkMode === 'edit' && originalImage) ? '' : 'none';
        // Sub-mode toggle (Default / Face Swap) only for Generate mode
        const submodeSection = document.getElementById('darkGenSubmodeSection');
        if (submodeSection) submodeSection.style.display = darkMode === 'generate' ? '' : 'none';
        // Face upload only for Face Swap sub-mode
        const faceSection = document.getElementById('darkFaceUploadSection');
        if (faceSection) faceSection.style.display = (darkMode === 'generate' && darkGenSubmode === 'faceswap') ? '' : 'none';
        // LoRA strength only for Generate + Default sub-mode
        const loraSection = document.getElementById('darkLoraStrengthSection');
        if (loraSection) loraSection.style.display = (darkMode === 'generate' && darkGenSubmode === 'default') ? '' : 'none';
        // Resolution only for Generate + Default sub-mode (BFS has fixed resolution)
        const resSection2 = document.getElementById('darkResolutionSection');
        if (resSection2) resSection2.style.display = (darkMode === 'generate' && darkGenSubmode === 'default') ? '' : 'none';
    }

    // Show/hide shared sections (Prompt + Generate) when image is loaded
    const hasContent = originalImage || darkGenNoImage;
    document.getElementById('promptSection').style.display = hasContent ? '' : 'none';
    document.getElementById('generateSection').style.display = hasContent ? '' : 'none';

    // In Dark Generate mode, show prompt + generate even without image
    if (darkGenNoImage) {
        document.getElementById('promptSection').style.display = '';
        document.getElementById('generateSection').style.display = '';
        const settingsSection = document.getElementById('darkSettingsSection');
        if (settingsSection) settingsSection.style.display = 'none';
        const resSection = document.getElementById('darkResolutionSection');
        if (resSection) resSection.style.display = '';
    }

    // Hide negative prompt in Dark Generate mode (ConditioningZeroOut handles it)
    if (mode === 'dark' && darkMode === 'generate') {
        const negSection = document.querySelector('.section-title.sub');
        const negInput = document.getElementById('negativeInput');
        if (negSection) negSection.style.display = 'none';
        if (negInput) { negInput.style.display = 'none'; negInput.value = ''; }
    } else {
        const negSection = document.querySelector('.section-title.sub');
        const negInput = document.getElementById('negativeInput');
        if (negSection) negSection.style.display = '';
        if (negInput) negInput.style.display = '';
    }

    // Update prompt placeholder
    const promptInput = document.getElementById('promptInput');
    if (mode === 'inpaint') {
        promptInput.placeholder = 'Describe what to paint in the mask...';
    } else if (mode === 'video') {
        promptInput.placeholder = 'Describe motion (woman slowly turns her head, smiles...)';
    } else if (mode === 'dark') {
        promptInput.placeholder = darkMode === 'generate'
            ? 'Describe the scene (misu, sitting on a couch, bedroom...)'
            : 'Dark Beast: describe NSFW edit...';
    } else {
        promptInput.placeholder = editSubmode === 'default'
            ? 'Describe the edit (put face from ref on t-shirt, change background...)'
            : 'Describe the edit (add cum on face, finger in ass, remove clothes...)';
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
    if (mode === 'inpaint') btnText.textContent = 'GENERATE';
    else if (mode === 'video') btnText.textContent = 'GENERATE VIDEO';
    else if (mode === 'dark') btnText.textContent = darkMode === 'generate' ? 'GENERATE' : 'DARK BEAST';
    else btnText.textContent = editSubmode === 'default' ? 'EDIT IMAGE' : 'DEPTH EDIT';

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

            // Bug #1: Show remove button
            const removeBtn = document.getElementById('removeMainPhoto');
            if (removeBtn) removeBtn.style.display = '';

            // Setup canvases for inpaint mode
            setupCanvases(img);

            // Refresh full UI (shows prompt, settings, edit submode toggle, etc.)
            switchMode(currentMode);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Bug #1: Remove main photo handler
function removeMainPhoto() {
    originalImage = null;
    const previewImg = uploadArea.querySelector('img');
    if (previewImg) previewImg.remove();
    uploadPlaceholder.style.display = '';
    const removeBtn = document.getElementById('removeMainPhoto');
    if (removeBtn) removeBtn.style.display = 'none';
    fileInput.value = '';

    // Hide all sections that depend on image
    document.getElementById('canvasSection').style.display = 'none';
    document.getElementById('promptSection').style.display = 'none';
    document.getElementById('settingsSection').style.display = 'none';
    document.getElementById('generateSection').style.display = 'none';
    document.querySelectorAll('.video-only, .inpaint-only, .image-only, .dark-only').forEach(el => {
        el.style.display = 'none';
    });

    // Re-show Dark mode/quality toggles if in dark mode
    if (currentMode === 'dark') {
        switchMode('dark');
    }
}

const removeMainPhotoBtn = document.getElementById('removeMainPhoto');
if (removeMainPhotoBtn) {
    removeMainPhotoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeMainPhoto();
    });
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

// Dark LoRA strength slider
const darkLoraStrengthSlider = document.getElementById('darkLoraStrengthSlider');
const darkLoraStrengthLabel = document.getElementById('darkLoraStrengthLabel');
if (darkLoraStrengthSlider) {
    darkLoraStrengthSlider.addEventListener('input', (e) => {
        darkLoraStrengthLabel.textContent = parseFloat(e.target.value).toFixed(2);
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

        // Show/hide resolution section (only for Generate + Default sub-mode)
        const resSection = document.getElementById('darkResolutionSection');
        if (resSection) resSection.style.display = (darkMode === 'generate' && darkGenSubmode === 'default') ? '' : 'none';

        // Show/hide LoRA strength section (only for Generate + Default sub-mode)
        const loraSection = document.getElementById('darkLoraStrengthSection');
        if (loraSection) loraSection.style.display = (darkMode === 'generate' && darkGenSubmode === 'default') ? '' : 'none';

        // Show/hide sub-mode section (only for Generate)
        const submodeSection = document.getElementById('darkGenSubmodeSection');
        if (submodeSection) submodeSection.style.display = darkMode === 'generate' ? '' : 'none';

        // Show/hide face upload (only for Generate + Face Swap)
        const faceSection = document.getElementById('darkFaceUploadSection');
        if (faceSection) faceSection.style.display = (darkMode === 'generate' && darkGenSubmode === 'faceswap') ? '' : 'none';

        // Update reference hint
        const hint = document.getElementById('darkImg2Hint');
        if (hint) {
            hint.textContent = darkMode === 'generate' ? '(optional: pose, object, prop)' : '(combine two girls)';
        }

        // Update settings visibility — hide denoise/steps for Generate (handled internally)
        const settingsSection = document.getElementById('darkSettingsSection');
        if (settingsSection) settingsSection.style.display = darkMode === 'edit' ? '' : 'none';

        // Refresh full UI (prompt placeholder, button text, etc.)
        switchMode('dark');
    });
});

// Dark resolution toggle
document.querySelectorAll('#darkResolutionSection .quality-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#darkResolutionSection .quality-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        darkResolution = btn.dataset.res;
    });
});

// Dark Generate sub-mode toggle (Default / Face Swap)
document.querySelectorAll('#darkGenSubmodeSection .quality-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#darkGenSubmodeSection .quality-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        darkGenSubmode = btn.dataset.gensubmode;

        // Toggle face upload and resolution/LoRA visibility
        const faceSection = document.getElementById('darkFaceUploadSection');
        if (faceSection) faceSection.style.display = darkGenSubmode === 'faceswap' ? '' : 'none';
        const loraSection = document.getElementById('darkLoraStrengthSection');
        if (loraSection) loraSection.style.display = darkGenSubmode === 'default' ? '' : 'none';
        const resSection = document.getElementById('darkResolutionSection');
        if (resSection) resSection.style.display = darkGenSubmode === 'default' ? '' : 'none';
    });
});

// Face photo upload for BFS face swap
const faceUploadArea = document.getElementById('faceUploadArea');
const faceFileInput = document.getElementById('faceFileInput');
const faceUploadPlaceholder = document.getElementById('faceUploadPlaceholder');

if (faceUploadArea && faceFileInput) {
    faceUploadArea.addEventListener('click', () => faceFileInput.click());
    faceFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            faceImageB64 = ev.target.result.split(',')[1]; // store raw base64
            // Show preview
            faceUploadPlaceholder.style.display = 'none';
            let preview = faceUploadArea.querySelector('img');
            if (!preview) {
                preview = document.createElement('img');
                preview.style.maxHeight = '80px';
                preview.style.borderRadius = '8px';
                faceUploadArea.appendChild(preview);
            }
            preview.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });
}

// ============================================================
// Dark Mode: Second Image Upload
// ============================================================
function loadDarkImage2(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new window.Image();
        img.onload = () => {
            darkImage2 = img;
            darkImg2Img.src = e.target.result;
            darkImg2Add.style.display = 'none';
            darkImg2Preview.style.display = '';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function removeDarkImage2() {
    darkImage2 = null;
    darkImg2Img.src = '';
    darkImg2Preview.style.display = 'none';
    darkImg2Add.style.display = '';
    if (darkFileInput2) darkFileInput2.value = '';
}

function getDarkImage2DataURL() {
    if (!darkImage2) return null;
    return smartImageToDataURL(darkImage2);
}

if (darkImg2Add) {
    darkImg2Add.addEventListener('click', () => {
        if (darkFileInput2) darkFileInput2.click();
    });

    darkImg2Add.addEventListener('dragover', (e) => {
        e.preventDefault();
        darkImg2Add.style.borderColor = 'rgba(255, 255, 255, 0.4)';
    });

    darkImg2Add.addEventListener('dragleave', () => {
        darkImg2Add.style.borderColor = '';
    });

    darkImg2Add.addEventListener('drop', (e) => {
        e.preventDefault();
        darkImg2Add.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadDarkImage2(file);
        }
    });
}

if (darkFileInput2) {
    darkFileInput2.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            loadDarkImage2(file);
        }
    });
}

if (darkImg2Remove) {
    darkImg2Remove.addEventListener('click', (e) => {
        e.stopPropagation();
        removeDarkImage2();
    });
}

// Ctrl+V paste support for dark image2 when the slot is visible
document.addEventListener('paste', (e) => {
    if (currentMode !== 'dark') return;
    // Only handle if we already have image1 (so paste goes to image2 slot)
    if (!originalImage) return;
    // If darkImage2 already set, don't override
    if (darkImage2) return;
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            const file = item.getAsFile();
            if (file) {
                loadDarkImage2(file);
                e.preventDefault();
            }
            break;
        }
    }
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
    return smartImageToDataURL(originalImage2);
}

// ============================================================
// Get mask as black/white image
// ============================================================
function getMaskDataURL() {
    // Export mask at the SAME resolution as smartImageToDataURL exports the image.
    // The canvas buffer may be smaller than the original image (scaled to fit the
    // container width), so we must scale the mask up to match the exported image.
    let targetW = originalImage.naturalWidth || originalImage.width;
    let targetH = originalImage.naturalHeight || originalImage.height;
    if (targetW > MAX_IMAGE_DIM || targetH > MAX_IMAGE_DIM) {
        const s = MAX_IMAGE_DIM / Math.max(targetW, targetH);
        targetW = Math.round(targetW * s);
        targetH = Math.round(targetH * s);
    }

    // Scale the drawn mask from canvas-buffer size → target image size
    const scaledMask = document.createElement('canvas');
    scaledMask.width = targetW;
    scaledMask.height = targetH;
    const sCtx = scaledMask.getContext('2d');
    sCtx.drawImage(maskCanvas, 0, 0, targetW, targetH);

    // Convert to B&W: painted areas (alpha > 10) → white, rest → black
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetW;
    tempCanvas.height = targetH;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#000';
    tempCtx.fillRect(0, 0, targetW, targetH);

    const maskData = sCtx.getImageData(0, 0, targetW, targetH);
    const tempData = tempCtx.getImageData(0, 0, targetW, targetH);

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

// Smart image export: resize to max 2048px, JPEG for smaller payloads
// Solves HEIC issues on iPhone/Mac and reduces file size ~10x vs PNG
const MAX_IMAGE_DIM = 2048;
const JPEG_QUALITY = 0.92;

function smartImageToDataURL(img) {
    let w = img.naturalWidth || img.width;
    let h = img.naturalHeight || img.height;

    // Resize if any dimension exceeds MAX
    if (w > MAX_IMAGE_DIM || h > MAX_IMAGE_DIM) {
        const scale = MAX_IMAGE_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
    }

    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    return c.toDataURL('image/jpeg', JPEG_QUALITY);
}

function getImageDataURL() {
    return smartImageToDataURL(originalImage);
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

    if (!originalImage && !(currentMode === 'dark' && darkMode === 'generate')) {
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
            progressText.textContent = 'Starting...';
        } else if (progress < 50) {
            progressText.textContent = 'Preparing...';
        } else {
            progressText.textContent = 'Generating...';
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
        progressText.textContent = 'Done';

        resetIphoneFilter();
        lastResultB64 = data.image;
        lastResultType = 'image';
        resultImage.src = 'data:image/png;base64,' + data.image;
        addToGallery(resultImage.src);
        resultImage.style.display = '';
        resultVideo.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        progressText.textContent = 'Error: ' + err.message;
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
let currentVideoController = null; // AbortController for cancel support
let currentVideoJobId = null; // RunPod job ID for polling
let videoPollingInterval = null; // Polling timer

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

    // Show cancel button
    const cancelBtn = document.getElementById('cancelVideoBtn');
    if (cancelBtn) cancelBtn.style.display = '';

    progressFill.style.width = '5%';
    progressText.textContent = 'Submitting job...';

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
            action: document.getElementById('actionSelect')?.value || 'none',
            shift: parseFloat(document.getElementById('shiftSlider')?.value || 5),
            cfg_high: parseFloat(document.getElementById('cfgHighSlider')?.value || 5),
            cfg_low: parseFloat(document.getElementById('cfgLowSlider')?.value || 1),
            lora_strength: parseFloat(document.getElementById('loraStrSlider')?.value || 1.3),
            scheduler: 'beta',
            video_steps: parseInt(document.getElementById('stepsSlider')?.value || 20),
        };

        if (audioEnabled) {
            body.audio_prompt = audioPrompt;
            body.audio_negative = audioNegative;
        }

        // Step 1: Submit job (fast — returns immediately)
        const submitResp = await fetch('/api/video', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });

        if (!submitResp.ok) {
            const errData = await submitResp.json().catch(() => null);
            throw new Error(errData?.error || `Server error: ${submitResp.status}`);
        }

        const submitData = await submitResp.json();
        if (!submitData.job_id) {
            throw new Error('No job_id in response');
        }

        currentVideoJobId = submitData.job_id;
        console.log('Video job submitted:', currentVideoJobId);
        progressFill.style.width = '10%';
        progressText.textContent = 'Job queued...';

        // Step 2: Poll for status every 3 seconds
        const videoDuration = frames / fps;
        let pollCount = 0;
        const maxPolls = 400; // 400 * 3s = 20 min max

        const result = await new Promise((resolve, reject) => {
            videoPollingInterval = setInterval(async () => {
                pollCount++;

                if (pollCount > maxPolls) {
                    clearInterval(videoPollingInterval);
                    videoPollingInterval = null;
                    reject(new Error('Video generation timed out (20 min)'));
                    return;
                }

                try {
                    const statusResp = await fetch(`/api/video/status/${currentVideoJobId}`, {
                        headers: authHeaders(),
                    });

                    if (!statusResp.ok) {
                        console.warn('Status poll error:', statusResp.status);
                        return; // retry on next poll
                    }

                    const statusData = await statusResp.json();
                    console.log('Poll #' + pollCount + ':', statusData.status);

                    if (statusData.status === 'IN_QUEUE') {
                        progressFill.style.width = '15%';
                        progressText.textContent = 'In queue...';
                    } else if (statusData.status === 'IN_PROGRESS') {
                        // Gradually fill progress bar during generation
                        const progress = Math.min(15 + (pollCount * 2), 85);
                        progressFill.style.width = progress + '%';
                        if (progress < 50) {
                            progressText.textContent = `Generating video (${frames} frames, ~${videoDuration.toFixed(1)}s)...`;
                        } else if (audioEnabled && progress > 70) {
                            progressText.textContent = 'Generating audio...';
                        } else {
                            progressText.textContent = `Generating... ${Math.round(progress)}%`;
                        }
                    } else if (statusData.status === 'COMPLETED') {
                        clearInterval(videoPollingInterval);
                        videoPollingInterval = null;
                        resolve(statusData);
                    } else if (statusData.status === 'FAILED') {
                        clearInterval(videoPollingInterval);
                        videoPollingInterval = null;
                        reject(new Error(statusData.error || 'Video generation failed'));
                    } else if (statusData.status === 'CANCELLED') {
                        clearInterval(videoPollingInterval);
                        videoPollingInterval = null;
                        reject(new Error('Video generation was cancelled'));
                    }
                } catch (pollErr) {
                    console.warn('Poll error:', pollErr);
                    // Don't reject — just retry on next interval
                }
            }, 3000);
        });

        // Step 3: Display video
        progressFill.style.width = '95%';
        progressText.textContent = 'Loading video...';

        console.log('Video completed. Base64 length:', result.video?.length || 0);
        lastResultB64 = result.video;
        lastResultType = 'video';
        const videoBlob = base64ToBlob(result.video, 'video/mp4');
        console.log('Video blob size:', videoBlob.size);
        const videoUrl = URL.createObjectURL(videoBlob);
        resultVideo.src = videoUrl;
        resultVideo.style.display = '';
        resultImage.style.display = 'none';
        resultSection.style.display = '';

        progressFill.style.width = '100%';
        progressText.textContent = 'Video ready';
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        if (videoPollingInterval) {
            clearInterval(videoPollingInterval);
            videoPollingInterval = null;
        }
        progressFill.style.width = '0%';
        const msg = err.name === 'AbortError' ? 'Request cancelled or timed out' : err.message;
        progressText.textContent = 'Error: ' + msg;
        if (err.name !== 'AbortError') alert('Error: ' + msg);
    } finally {
        currentVideoController = null;
        currentVideoJobId = null;
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';
    }
}

async function cancelVideoGeneration() {
    // Stop polling
    if (videoPollingInterval) {
        clearInterval(videoPollingInterval);
        videoPollingInterval = null;
    }

    // Cancel RunPod job
    if (currentVideoJobId) {
        progressText.textContent = 'Cancelling...';
        try {
            await fetch(`/api/video/cancel/${currentVideoJobId}`, {
                method: 'POST',
                headers: authHeaders(),
            });
        } catch (e) {
            console.warn('Cancel request failed:', e);
        }
        currentVideoJobId = null;
    }

    // Abort any pending fetch
    if (currentVideoController) {
        currentVideoController.abort();
        currentVideoController = null;
    }

    progressText.textContent = 'Cancelled';
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
    progressText.textContent = 'Sending...';
    resultSection.style.display = 'none';

    let progressInterval;

    try {
        progressInterval = setInterval(() => {
            const cur = parseFloat(progressFill.style.width);
            if (cur < 85) {
                progressFill.style.width = (cur + 1.5) + '%';
                if (cur > 30) progressText.textContent = 'Generating...';
                if (cur > 70) progressText.textContent = 'Almost done...';
            }
        }, 1500);

        const body = {
            image: imageDataURL.split(',')[1],
            prompt: prompt,
            negative: negative,
            denoise: denoise,
            steps: editSubmode === 'default' ? 4 : steps,
            edit_submode: editSubmode,
        };

        // LoRA (optional)
        const loraName = document.getElementById('loraNameInput')?.value?.trim() || '';
        const loraStrength = parseFloat(document.getElementById('loraStrengthSlider')?.value || 1.0);
        if (loraName) {
            body.lora_name = loraName;
            body.lora_strength = loraStrength;
        }

        // Include reference image (from Edit image2 area OR dark image2 area)
        if (image2DataURL) {
            body.image2 = image2DataURL.split(',')[1];
        } else {
            const darkImg2 = getDarkImage2DataURL();
            if (darkImg2) body.image2 = darkImg2.split(',')[1];
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
        progressText.textContent = 'Done';

        resetIphoneFilter();
        lastResultB64 = data.image;
        lastResultType = 'image';
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
        progressText.textContent = 'Error: ' + err.message;
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

    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    progressFill.style.width = '10%';
    progressText.textContent = 'Processing...';
    resultSection.style.display = 'none';

    // Show cancel button
    const cancelBtn = document.getElementById('cancelVideoBtn');
    if (cancelBtn) cancelBtn.style.display = '';

    let progressInterval;

    try {
        progressInterval = setInterval(() => {
            const cur = parseFloat(progressFill.style.width);
            if (cur < 85) {
                progressFill.style.width = (cur + 1.5) + '%';
                if (cur > 30) progressText.textContent = 'Generating...';
                if (cur > 70) progressText.textContent = 'Almost done...';
            }
        }, 1500);

        const body = {
            prompt: prompt,
            negative: negative,
            quality: darkQuality,
            mode: darkMode,
        };

        if (darkMode === 'generate') {
            body.submode = darkGenSubmode;

            if (darkGenSubmode === 'faceswap') {
                // BFS Face Swap: face image is required
                if (!faceImageB64) {
                    throw new Error('Upload a face photo first');
                }
                body.face_image = faceImageB64;
                body.image = ''; // no main image needed
            } else {
                // Default: text2img with optional reference
                const [w, h] = darkResolution.split('x').map(Number);
                body.width = w;
                body.height = h;
                const loraSlider = document.getElementById('darkLoraStrengthSlider');
                if (loraSlider) body.lora_strength = parseFloat(loraSlider.value);
                if (originalImage) {
                    const imageDataURL = getImageDataURL();
                    body.image = imageDataURL.split(',')[1];
                } else {
                    body.image = '';
                }
            }
        } else {
            // Edit mode: image is required
            const imageDataURL = getImageDataURL();
            body.image = imageDataURL.split(',')[1];
            body.denoise = denoise;
            body.steps = steps;
            const darkImg2URL = getDarkImage2DataURL();
            if (darkImg2URL) {
                body.image2 = darkImg2URL.split(',')[1];
            }
        }

        currentVideoController = new AbortController();
        const fetchTimeout = setTimeout(() => currentVideoController.abort(), 10 * 60 * 1000); // 10 min

        const response = await fetch('/api/image-edit-dark', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
            signal: currentVideoController.signal,
        });

        clearTimeout(fetchTimeout);
        clearInterval(progressInterval);

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        progressFill.style.width = '100%';
        progressText.textContent = 'Done';

        resetIphoneFilter();
        lastResultB64 = data.image;
        lastResultType = 'image';
        resultImage.src = 'data:image/png;base64,' + data.image;
        addToGallery(resultImage.src);
        resultImage.style.display = '';
        resultVideo.style.display = 'none';
        resultSection.style.display = '';

        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        clearInterval(progressInterval);
        progressFill.style.width = '0%';
        const msg = err.name === 'AbortError' ? 'Request cancelled' : err.message;
        progressText.textContent = 'Error: ' + msg;
        if (err.name !== 'AbortError') alert('Error: ' + msg);
    } finally {
        currentVideoController = null;
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';
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
// Download, Copy & Retry  (iOS / Safari / Telegram WebView safe)
// ============================================================

// Helper: data-URL → Blob
function dataURLtoBlob(dataUrl) {
    const [header, b64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
}

// iOS-safe download: open in new tab so user can long-press → Save
function triggerDownload(dataUrl, filename) {
    // Try the standard <a download> first (works on desktop & Android)
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // On iOS WebView <a download> silently fails — also open in new tab
    const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (isIOS) {
        // Convert to object URL so Safari can display it natively
        const blob = dataURLtoBlob(dataUrl);
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
        // Clean up after a delay
        setTimeout(() => URL.revokeObjectURL(url), 60000);
    }
}

downloadBtn.addEventListener('click', () => {
    if ((currentMode === 'inpaint' || currentMode === 'image' || currentMode === 'dark') && resultImage.src) {
        const names = { inpaint: 'inpaint_result.png', image: 'edit_result.png', dark: 'dark_result.png' };
        triggerDownload(resultImage.src, names[currentMode] || 'result.png');
    } else if (currentMode === 'video' && resultVideo.src) {
        triggerDownload(resultVideo.src, 'video_result.mp4');
    }
});

// iOS-safe copy: convert to PNG blob via canvas, then Clipboard API
async function copyImageToClipboard(imgSrc) {
    try {
        // Draw image to canvas to get a clean PNG blob (required by ClipboardItem)
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = imgSrc;
        });
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d').drawImage(img, 0, 0);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        // Safari requires the ClipboardItem Promise pattern
        if (navigator.clipboard && typeof ClipboardItem !== 'undefined') {
            await navigator.clipboard.write([
                new ClipboardItem({ 'image/png': blob })
            ]);
            return true;
        }

        // Fallback: open in new tab
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        return true;
    } catch (e) {
        console.error('Copy failed:', e);
        // Last resort fallback: open image in new tab
        try {
            window.open(imgSrc, '_blank');
            return true;
        } catch (_) { }
        return false;
    }
}

const copyBtn = document.getElementById('copyBtn');
if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
        if (resultImage.src) {
            const ok = await copyImageToClipboard(resultImage.src);
            copyBtn.textContent = ok ? 'Copied' : 'Opened';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
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
// iPhone Camera Filter
// ============================================================
let iphoneFilterActive = false;
let originalResultSrc = null; // store original before filter
let filteredResultSrc = null; // cache filtered version
let lightboxIphoneActive = false;
let lightboxOriginalSrc = null;
let lightboxFilteredSrc = null;

const iphoneToggle = document.getElementById('iphoneToggle');
const lightboxIphoneToggle = document.getElementById('lightboxIphoneToggle');

/**
 * Apply iPhone camera filter to an image dataURL.
 * Calibrated to match iPhone 16 Pro computational photography:
 * - Apple warm color science: R×1.02, G×1.005, B×0.97
 * - Sensor noise: σ=2.8 (fine grain, mostly visible in shadows)
 * - Slight shadow lift (HDR-like): +8 on dark pixels
 * - Micro-contrast reduction (Deep Fusion smoothing): gentle blur-like averaging
 * - JPEG quality 82 (Apple's default for non-ProRAW)
 */
function applyIphoneFilter(dataUrl) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            const len = data.length;

            for (let i = 0; i < len; i += 4) {
                let r = data[i];
                let g = data[i + 1];
                let b = data[i + 2];

                // 1. Apple warm color science
                r = Math.min(255, r * 1.02);
                g = Math.min(255, g * 1.005);
                b = Math.min(255, b * 0.97);

                // 2. Shadow lift (iPhone HDR: brightens dark areas slightly)
                if (r < 60) r += 8;
                if (g < 60) g += 8;
                if (b < 60) b += 6;

                // 3. Subtle saturation boost (Apple's vibrant processing)
                const avg = (r + g + b) / 3;
                r = r + (r - avg) * 0.06;
                g = g + (g - avg) * 0.06;
                b = b + (b - avg) * 0.06;

                // 4. Fine sensor noise (σ=2.8, shadow-weighted)
                const luminance = (r * 0.299 + g * 0.587 + b * 0.114);
                // More noise in shadows (like real sensor behavior)
                const noiseMult = luminance < 80 ? 1.4 : (luminance < 160 ? 1.0 : 0.6);
                const noise = (Math.random() - 0.5) * 5.6 * noiseMult;
                r += noise;
                g += noise;
                b += noise;

                data[i] = Math.max(0, Math.min(255, r));
                data[i + 1] = Math.max(0, Math.min(255, g));
                data[i + 2] = Math.max(0, Math.min(255, b));
            }

            ctx.putImageData(imageData, 0, 0);

            // 5. JPEG compression (quality 82 — Apple's default)
            resolve(canvas.toDataURL('image/jpeg', 0.82));
        };
        img.src = dataUrl;
    });
}

// --- Result section iPhone toggle ---
if (iphoneToggle) {
    iphoneToggle.addEventListener('click', async () => {
        if (!resultImage.src || resultImage.style.display === 'none') return;

        if (!iphoneFilterActive) {
            // Apply filter
            iphoneToggle.textContent = '⏳...';
            iphoneToggle.disabled = true;
            originalResultSrc = originalResultSrc || resultImage.src;
            if (!filteredResultSrc) {
                filteredResultSrc = await applyIphoneFilter(originalResultSrc);
            }
            resultImage.src = filteredResultSrc;
            iphoneToggle.classList.add('active');
            iphoneToggle.textContent = '📱 iPhone';
            iphoneToggle.disabled = false;
            iphoneFilterActive = true;
        } else {
            // Revert to original
            resultImage.src = originalResultSrc;
            iphoneToggle.classList.remove('active');
            iphoneFilterActive = false;
        }
    });
}

// --- Lightbox iPhone toggle ---
if (lightboxIphoneToggle) {
    lightboxIphoneToggle.addEventListener('click', async () => {
        if (activeLightboxIndex < 0) return;

        if (!lightboxIphoneActive) {
            lightboxIphoneToggle.textContent = '⏳...';
            lightboxIphoneToggle.disabled = true;
            lightboxOriginalSrc = lightboxOriginalSrc || galleryItems[activeLightboxIndex].dataUrl;
            if (!lightboxFilteredSrc) {
                lightboxFilteredSrc = await applyIphoneFilter(lightboxOriginalSrc);
            }
            lightboxImage.src = lightboxFilteredSrc;
            lightboxIphoneToggle.classList.add('active');
            lightboxIphoneToggle.textContent = '📱 iPhone';
            lightboxIphoneToggle.disabled = false;
            lightboxIphoneActive = true;
        } else {
            lightboxImage.src = lightboxOriginalSrc;
            lightboxIphoneToggle.classList.remove('active');
            lightboxIphoneActive = false;
        }
    });
}

// Override lightbox Save/Copy/Send to use filtered version when active
const _origLightboxDownloadHandler = lightboxDownload?.onclick;

// When switching lightbox images, reset filter state
const _origOpenLightbox = openLightbox;

// Reset iPhone filter state when new result is generated or lightbox changes
function resetIphoneFilter() {
    iphoneFilterActive = false;
    originalResultSrc = null;
    filteredResultSrc = null;
    if (iphoneToggle) iphoneToggle.classList.remove('active');
}

function resetLightboxIphoneFilter() {
    lightboxIphoneActive = false;
    lightboxOriginalSrc = null;
    lightboxFilteredSrc = null;
    if (lightboxIphoneToggle) lightboxIphoneToggle.classList.remove('active');
}

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
    resetLightboxIphoneFilter();
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
        const src = lightboxIphoneActive ? lightboxFilteredSrc : galleryItems[activeLightboxIndex].dataUrl;
        const ext = lightboxIphoneActive ? 'jpg' : 'png';
        triggerDownload(src, `gallery_${activeLightboxIndex + 1}.${ext}`);
    });
}

if (lightboxCopy) {
    lightboxCopy.addEventListener('click', async () => {
        if (activeLightboxIndex < 0) return;
        const src = lightboxIphoneActive ? lightboxFilteredSrc : galleryItems[activeLightboxIndex].dataUrl;
        const ok = await copyImageToClipboard(src);
        lightboxCopy.textContent = ok ? 'Copied' : 'Failed';
        setTimeout(() => { lightboxCopy.textContent = 'Copy'; }, 1500);
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

const lightboxSend = document.getElementById('lightboxSend');
if (lightboxSend) {
    lightboxSend.addEventListener('click', async () => {
        if (activeLightboxIndex < 0) return;
        const dataUrl = lightboxIphoneActive ? lightboxFilteredSrc : galleryItems[activeLightboxIndex].dataUrl;
        const mediaB64 = dataUrl.split(',')[1];
        if (!mediaB64) return;

        lightboxSend.textContent = 'Sending...';
        lightboxSend.disabled = true;

        try {
            const resp = await fetch('/api/send', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ media: mediaB64, type: 'image' }),
            });
            lightboxSend.textContent = resp.ok ? 'Sent' : 'Failed';
        } catch (e) {
            lightboxSend.textContent = 'Error';
            console.error('Lightbox send error:', e);
        }

        lightboxSend.disabled = false;
        setTimeout(() => { lightboxSend.textContent = 'Send'; }, 2000);
    });
}

// ============================================================
// Preset Prompt Buttons
// ============================================================
const PRESETS = {
    inpaint: [
        {
            label: 'Name',
            prompt: 'The word "Name" handwritten in dark crimson lipstick on bare skin. Semi-transparent smeared lipstick, skin texture visible through the letters. Faded messy imperfect handwritten letters, slightly uneven and crooked. Dark burgundy red lipstick lightly applied on skin. Photorealistic.',
            negative: 'blurry, ugly, deformed, font, low quality, cartoon',
        },
    ],
    dark: [
        {
            label: 'Cum',
            prompt: 'thick white cum dripping on skin, semen splattered, creampie leaking, wet glistening cum drops, realistic bodily fluid, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Pee',
            prompt: 'clear transparent stream of pee flowing between legs, warm liquid dripping on thighs, wet glistening skin, watersports, clear fluid like water, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Shit',
            prompt: 'brown feces smeared on skin, dirty messy scat, soiled body, realistic texture, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Nude',
            prompt: 'completely naked, fully nude, no clothes, bare breasts with erect nipples, exposed pussy, smooth skin, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, clothes, dressed, fabric',
        },
        {
            label: 'Anal',
            prompt: 'man\'s thick cock deep inside her ass, anal penetration from behind, stretched anus around penis, doggy style anal sex, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, extra limbs',
        },
    ],
    image: [
        {
            label: 'Cum',
            prompt: 'thick white cum dripping on skin, semen splattered, wet glistening cum drops, realistic bodily fluid, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Pee',
            prompt: 'clear transparent stream of pee flowing between legs, warm liquid dripping on thighs, wet glistening skin, watersports, clear fluid like water, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Shit',
            prompt: 'brown feces smeared on skin, dirty messy scat, soiled body, realistic texture, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy',
        },
        {
            label: 'Nude',
            prompt: 'completely naked, fully nude, no clothes, bare breasts with erect nipples, exposed pussy, smooth skin, photorealistic',
            negative: 'blurry, ugly, deformed, watermark, text, low quality, cartoon, bad anatomy, clothes, dressed, fabric',
        },
        {
            label: 'Anal',
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

// ============================================================
// Send to Telegram
// ============================================================
let lastResultB64 = null;
let lastResultType = 'image'; // 'image' or 'video'

const sendBtn = document.getElementById('sendBtn');
if (sendBtn) {
    sendBtn.addEventListener('click', async () => {
        let mediaB64, mediaType;

        if (resultVideo.style.display !== 'none' && resultVideo.src) {
            // For video, we need the raw base64 — stored from generation
            if (lastResultType === 'video' && lastResultB64) {
                mediaB64 = lastResultB64;
                mediaType = 'video';
            } else {
                sendBtn.textContent = 'No data';
                setTimeout(() => { sendBtn.textContent = 'Send'; }, 1500);
                return;
            }
        } else if (resultImage.src && resultImage.src.startsWith('data:')) {
            mediaB64 = resultImage.src.split(',')[1];
            mediaType = 'image';
        } else {
            sendBtn.textContent = 'No result';
            setTimeout(() => { sendBtn.textContent = 'Send'; }, 1500);
            return;
        }

        sendBtn.textContent = 'Sending...';
        sendBtn.disabled = true;

        try {
            const resp = await fetch('/api/send', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ media: mediaB64, type: mediaType }),
            });

            if (resp.ok) {
                sendBtn.textContent = 'Sent';
            } else {
                const err = await resp.json().catch(() => null);
                sendBtn.textContent = 'Failed';
                console.error('Send error:', err);
            }
        } catch (e) {
            sendBtn.textContent = 'Error';
            console.error('Send error:', e);
        }

        sendBtn.disabled = false;
        setTimeout(() => { sendBtn.textContent = 'Send'; }, 2000);
    });
}

// ============================================================
// All-In-One Trigger Words
// ============================================================
const actionSelect = document.getElementById('actionSelect');
const aioTriggerRow = document.getElementById('aioTriggerRow');

if (actionSelect && aioTriggerRow) {
    // Show/hide trigger row when action changes
    actionSelect.addEventListener('change', () => {
        aioTriggerRow.style.display = actionSelect.value === 'allinone' ? '' : 'none';
    });

    // Click trigger button → insert at start of prompt
    document.querySelectorAll('#aioTriggers .preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const trigger = btn.dataset.trigger;
            const promptInput = document.getElementById('promptInput');
            const current = promptInput.value.trim();
            // Replace existing trigger or prepend
            const allTriggers = ['m15510n4ry', 'bl0wj0b', 'd0ubl3_bj', 'c0wg1rl', 'd0gg1e'];
            let cleaned = current;
            allTriggers.forEach(t => {
                cleaned = cleaned.replace(new RegExp('\\b' + t + '\\b,?\\s*', 'g'), '');
            });
            promptInput.value = trigger + (cleaned ? ', ' + cleaned : '');

            // Highlight active trigger
            document.querySelectorAll('#aioTriggers .preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}
