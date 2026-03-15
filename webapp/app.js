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
    if (resp.status === 402) {
        alert('❌ Not enough tokens. Buy more in the Profile tab.');
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
let lastResultB64 = null;
let lastResultType = 'image'; // 'image' or 'video'

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

// Gallery & Lightbox
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

// Top Tabs
const topTabBtns = document.querySelectorAll('.top-tab');
const modeSelect = document.getElementById('modeSelect');
let currentTab = 'generate';

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
// Top Tab Navigation
// ============================================================
topTabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
    currentTab = tab;
    topTabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('contentFaq').style.display = tab === 'faq' ? '' : 'none';
    document.getElementById('contentGenerate').style.display = tab === 'generate' ? '' : 'none';
    document.getElementById('contentProfile').style.display = tab === 'profile' ? '' : 'none';
    if (tab === 'profile') loadProfile();
}

// ============================================================
// Profile + In-App Purchases (Telegram Stars)
// ============================================================
async function loadProfile() {
    try {
        const resp = await fetch('/api/auth', { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();

        document.getElementById('profileName').textContent =
            window.Telegram?.WebApp?.initDataUnsafe?.user?.first_name || 'User';
        document.getElementById('profileTokens').textContent = data.tokens ?? 0;
        document.getElementById('profileGens').textContent = data.total_gens ?? 0;

        const isPrem = data.is_premium;
        document.getElementById('profilePremium').textContent = isPrem ? 'Active' : 'None';
        document.getElementById('profileRole').textContent =
            data.is_admin ? 'Admin' : (isPrem ? 'Premium' : 'User');
    } catch (e) {
        console.error('loadProfile error:', e);
    }
}

async function buyTokens(packageId) {
    try {
        const resp = await fetch('/api/buy-tokens', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
            body: JSON.stringify({ package: packageId }),
        });
        const data = await resp.json();
        if (data.invoice_url && window.Telegram?.WebApp?.openInvoice) {
            window.Telegram.WebApp.openInvoice(data.invoice_url, (status) => {
                if (status === 'paid') loadProfile();
            });
        }
    } catch (e) {
        console.error('buyTokens error:', e);
    }
}

async function buyPremium() {
    try {
        const resp = await fetch('/api/buy-premium', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
        });
        const data = await resp.json();
        if (data.invoice_url && window.Telegram?.WebApp?.openInvoice) {
            window.Telegram.WebApp.openInvoice(data.invoice_url, (status) => {
                if (status === 'paid') loadProfile();
            });
        }
    } catch (e) {
        console.error('buyPremium error:', e);
    }
}

// Bind buy buttons
document.querySelectorAll('.buy-btn[data-package]').forEach(btn => {
    btn.addEventListener('click', () => buyTokens(btn.dataset.package));
});
document.getElementById('buyPremiumBtn')?.addEventListener('click', buyPremium);

// ============================================================
// Mode Dropdown (Custom)
// ============================================================
const customSelect = document.getElementById('customSelect');
const selectTrigger = document.getElementById('selectTrigger');
const selectText = document.getElementById('selectText');
const selectOptions = document.getElementById('selectOptions');

selectTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    customSelect.classList.toggle('open');
});

// Close dropdown when clicking outside
document.addEventListener('click', () => {
    customSelect.classList.remove('open');
});

selectOptions.addEventListener('click', (e) => {
    e.stopPropagation();
});

document.querySelectorAll('.custom-option').forEach(opt => {
    opt.addEventListener('click', () => {
        const val = opt.dataset.value;
        // Update visual state
        document.querySelectorAll('.custom-option').forEach(o => o.classList.remove('active'));
        opt.classList.add('active');
        selectText.textContent = opt.textContent;
        customSelect.classList.remove('open');

        // Sync hidden select
        modeSelect.value = val;

        // Trigger mode switch
        if (val === 'generate') {
            darkMode = 'generate';
            switchMode('dark');
        } else {
            if (val === 'dark') darkMode = 'edit';
            switchMode(val);
        }
    });
});

function switchMode(mode) {
    currentMode = mode;

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
        // Quality only for Edit mode
        const qualitySection = document.getElementById('darkQualitySection');
        if (qualitySection) qualitySection.style.display = darkMode === 'edit' ? '' : 'none';
        // NEVER show Reference image section in Dark mode (user request)
        const img2Section = document.getElementById('darkImage2Section');
        if (img2Section) img2Section.style.display = 'none';
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
    // Video mode hides prompt section (Kenpechi has per-scene prompts)
    const hasContent = originalImage || darkGenNoImage;
    document.getElementById('promptSection').style.display = (hasContent && mode !== 'video') ? '' : 'none';
    document.getElementById('generateSection').style.display = hasContent ? '' : 'none';

    // Hide batch count for video (only 1 generation at a time)
    const batchRow = document.querySelector('.batch-row');
    if (batchRow) batchRow.style.display = mode === 'video' ? 'none' : '';

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

if (framesSlider) {
    framesSlider.addEventListener('input', (e) => {
        framesLabel.textContent = e.target.value;
    });
}

// Audio toggle
if (audioToggle) {
    audioToggle.addEventListener('change', () => {
        audioSettings.style.display = audioToggle.checked ? '' : 'none';
    });
}

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

// (Dark edit/generate toggle removed — handled by mode dropdown now)

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

    // Video mode uses per-scene prompts, not the shared prompt field
    if (currentMode === 'video') {
        await generateKenpechiVideo();
        return;
    }

    if (!prompt) {
        alert('Enter a prompt!');
        return;
    }

    if (!originalImage && !(currentMode === 'dark' && darkMode === 'generate')) {
        alert('Upload a photo first!');
        return;
    }

    const count = batchCount;

    for (let i = 0; i < count; i++) {
        if (count > 1) {
            progressInfo.style.display = '';
            progressText.textContent = `🔄 Generating ${i + 1}/${count}...`;
        }

        if (currentMode === 'inpaint') {
            await generateInpaint(prompt);
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

        if (handleAuthError(response)) {
            progressFill.style.width = '0%';
            progressText.textContent = '';
            return;
        }
        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.error || `Server error: ${response.status}`);
        }

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

        if (handleAuthError(response)) {
            progressFill.style.width = '0%';
            progressText.textContent = '';
            return;
        }
        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.error || `Server error: ${response.status}`);
        }

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
            iphoneToggle.textContent = 'iPhone';
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
            lightboxIphoneToggle.textContent = 'iPhone';
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

// ============================================================
// Profile & History
// ============================================================
const ADMIN_ID = 1946394239;

function getTgUser() {
    try {
        const user = window.Telegram?.WebApp?.initDataUnsafe?.user;
        return user || null;
    } catch (_) { return null; }
}

async function loadProfile() {
    const user = getTgUser();
    const nameEl = document.getElementById('profileName');
    const idEl = document.getElementById('profileId');
    const badgeEl = document.getElementById('profileBadge');
    const tokenEl = document.getElementById('tokenCount');
    const premDesc = document.getElementById('premiumDesc');
    const premBtn = document.getElementById('buyPremiumBtn');
    const premCard = document.getElementById('premiumCard');

    // Set basic info from Telegram
    if (user) {
        nameEl.textContent = user.first_name || user.username || 'User';
        idEl.textContent = user.id;
    }

    // Fetch profile from API
    try {
        const resp = await fetch('/api/profile', { headers: authHeaders() });
        if (resp.ok) {
            const data = await resp.json();
            tokenEl.textContent = data.tokens ?? 0;

            // Badge
            if (data.is_admin) {
                badgeEl.textContent = 'Admin';
                badgeEl.className = 'profile-badge admin';
                // Admin has unlimited — hide premium buy
                premCard.classList.add('active-premium');
                premDesc.textContent = 'Unlimited (Admin)';
                premBtn.textContent = '✓ Admin';
                premBtn.disabled = true;
            } else if (data.is_premium) {
                badgeEl.textContent = 'Premium';
                badgeEl.className = 'profile-badge premium';
                premCard.classList.add('active-premium');
                premDesc.textContent = data.premium_until
                    ? 'Active until ' + new Date(data.premium_until).toLocaleDateString()
                    : 'Unlimited generations & priority queue';
                premBtn.textContent = '✓ Active';
                premBtn.disabled = true;
            } else {
                badgeEl.textContent = 'Free';
                badgeEl.className = 'profile-badge free';
            }
        }
    } catch (e) {
        console.error('Profile load error:', e);
    }

    // Load history
    loadHistory();
}

async function loadHistory() {
    const historyList = document.getElementById('historyList');
    try {
        const resp = await fetch('/api/history?limit=20', { headers: authHeaders() });
        if (!resp.ok) return;
        const items = await resp.json();

        if (!items.length) {
            historyList.innerHTML = '<div class="history-empty">No generations yet</div>';
            return;
        }

        historyList.innerHTML = items.map(item => `
            <div class="history-item" data-id="${item.id}">
                <div class="history-prompt">${escapeHtml(item.prompt)}</div>
                <div class="history-meta">
                    <span class="history-date">${formatDate(item.created_at)}</span>
                    <span class="history-mode">${item.mode || 'generate'}</span>
                </div>
                <button class="history-delete" onclick="deleteHistory(${item.id})" title="Delete">✕</button>
            </div>
        `).join('');
    } catch (e) {
        console.error('History load error:', e);
    }
}

async function deleteHistory(id) {
    try {
        const resp = await fetch('/api/history/' + id, {
            method: 'DELETE',
            headers: authHeaders(),
        });
        if (resp.ok) {
            const el = document.querySelector(`.history-item[data-id="${id}"]`);
            if (el) {
                el.style.opacity = '0';
                el.style.transform = 'translateX(20px)';
                setTimeout(() => el.remove(), 300);
            }
        }
    } catch (e) {
        console.error('Delete error:', e);
    }
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// ============================================================
// Buy Tokens & Premium
// ============================================================
document.querySelectorAll('.token-buy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
        const packageId = btn.dataset.package;
        btn.classList.add('buying');
        try {
            const resp = await fetch('/api/buy-tokens', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ package: packageId }),
            });
            if (resp.ok) {
                // Invoice sent to chat — close WebApp so user sees it
                if (window.Telegram?.WebApp) {
                    window.Telegram.WebApp.close();
                }
            } else {
                btn.classList.remove('buying');
            }
        } catch (e) {
            console.error('Buy error:', e);
            btn.classList.remove('buying');
        }
    });
});

const premiumBtn = document.getElementById('buyPremiumBtn');
if (premiumBtn) {
    premiumBtn.addEventListener('click', async () => {
        premiumBtn.disabled = true;
        premiumBtn.textContent = 'Sending...';
        try {
            const resp = await fetch('/api/buy-premium', {
                method: 'POST',
                headers: authHeaders(),
            });
            if (resp.ok) {
                if (window.Telegram?.WebApp) {
                    window.Telegram.WebApp.close();
                }
            } else {
                premiumBtn.disabled = false;
                premiumBtn.textContent = 'Get Premium — 1500 Stars';
            }
        } catch (e) {
            console.error('Premium buy error:', e);
            premiumBtn.disabled = false;
            premiumBtn.textContent = 'Get Premium — 1500 Stars';
        }
    });
}

// ============================================================
// Kenpechi SVI — 6-Scene Multi-Pass Video
// ============================================================

// Available LoRAs (pairs: HIGH and LOW variants)
const KENPECHI_LORAS_HIGH = [
    'HIGH/DR34ML4Y_HIGH_V2.safetensors',
    'HIGH/NSFW-22-H-e8.safetensors',
    'HIGH/Deepthroat-W22-I2V-HN.safetensors',
    'HIGH/F4c3spl4sh-high-k3nk.safetensors',
    'HIGH/Oral-insertion-high-v1.0.safetensors',
    // NSFW_pack
    'NSFW_pack/WAN-2.2-I2V-Double-Blowjob-HIGH-v1.safetensors',
    'NSFW_pack/WAN-2.2-I2V-Body-Cumshot-HIGH-v1.safetensors',
    'NSFW_pack/wan22-mouthfull-140epoc-high-k3nk.safetensors',
    'NSFW_pack/Blink_Squatting_Cowgirl_Position_I2V_HIGH.safetensors',
    'NSFW_pack/PENISLORA_22_i2v_HIGH_e320.safetensors',
    'NSFW_pack/Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_high_noise.safetensors',
    'NSFW_pack/W22_Multiscene_Photoshoot_Softcore_i2v_HN.safetensors',
    'NSFW_pack/WAN-2.2-I2V-HandjobBlowjobCombo-HIGH-v1.safetensors',
    'NSFW_pack/WAN-2.2-I2V-SensualTeasingBlowjob-HIGH-v1.safetensors',
    'NSFW_pack/iGOON_Blink_Blowjob_I2V_HIGH.safetensors',
    'NSFW_pack/iGoon - Blink_Front_Doggystyle_I2V_HIGH.safetensors',
    'NSFW_pack/iGoon - Blink_Missionary_I2V_HIGH.safetensors',
    'NSFW_pack/iGoon - Blink_Back_Doggystyle_HIGH.safetensors',
    'NSFW_pack/iGoon - Blink_Facial_I2V_HIGH.safetensors',
    'NSFW_pack/iGoon_Blink_Missionary_I2V_HIGH v2.safetensors',
    'NSFW_pack/iGoon_Blink_Titjob_I2V_HIGH.safetensors',
    'NSFW_pack/lips-bj_high_noise.safetensors',
    'NSFW_pack/mql_casting_sex_doggy_kneel_diagonally_behind_vagina_wan22_i2v_v1_high_noise.safetensors',
    'NSFW_pack/mql_casting_sex_reverse_cowgirl_lie_front_vagina_wan22_i2v_v1_high_noise.safetensors',
    'NSFW_pack/mql_casting_sex_spoon_wan22_i2v_v1_high_noise.safetensors',
    'NSFW_pack/mql_doggy_a_wan22_t2v_v1_high_noise .safetensors',
    'NSFW_pack/mql_massage_tits_wan22_i2v_v1_high_noise.safetensors',
    'NSFW_pack/mql_panties_aside_wan22_i2v_v1_high_noise.safetensors',
    'NSFW_pack/mqlspn_a_wan22_t2v_v1_high_noise.safetensors',
    'NSFW_pack/sfbehind_v2.1_high_noise.safetensors',
    'NSFW_pack/sid3l3g_transition_v2.0_H.safetensors',
    'NSFW_pack/wan2.2_i2v_high_ulitmate_pussy_asshole.safetensors',
    // Root loras
    'erect_penis_epoch_80.safetensors',
    'pov-blowjob-i2v-v1.2.safetensors',
    'jfj-deepthroat-W22-I2V-HN.safetensors',
    'front_doggy_plow_v1_1_wan.safetensors',
    'pov-missionary-i2v-high-v1.0.safetensors',
    'side-sex-i2v-v10.safetensors',
    'wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors',
    'fingering-high-v1.0.safetensors',
    'nipple_stroke_WAN22_I2V_v1_high_noise.safetensors',
];

const KENPECHI_LORAS_LOW = [
    'LOW/DR34ML4Y_LOW_V2.safetensors',
    'LOW/NSFW-22-L-e8.safetensors',
    'LOW/Deepthroat-W22-I2V-LN.safetensors',
    'LOW/F4c3spl4sh-low-k3nk.safetensors',
    'LOW/Oral-insertion-low-v1.0.safetensors',
    // NSFW_pack
    'NSFW_pack/WAN-2.2-I2V-Double-Blowjob-LOW-v1.safetensors',
    'NSFW_pack/WAN-2.2-I2V-Body-Cumshot-LOW-v1.safetensors',
    'NSFW_pack/wan22-mouthfull-152epoc-low-k3nk.safetensors',
    'NSFW_pack/Blink_Squatting_Cowgirl_Position_I2V_LOW.safetensors',
    'NSFW_pack/PENISLORA_22_i2v_LOW_e496.safetensors',
    'NSFW_pack/Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_low_noise.safetensors',
    'NSFW_pack/W22_Multiscene_Photoshoot_Softcore_i2v_LN.safetensors',
    'NSFW_pack/WAN-2.2-I2V-HandjobBlowjobCombo-LOW-v1.safetensors',
    'NSFW_pack/WAN-2.2-I2V-SensualTeasingBlowjob-LOW-v1.safetensors',
    'NSFW_pack/iGOON_Blink_Blowjob_I2V_LOW.safetensors',
    'NSFW_pack/iGoon - Blink_Front_Doggystyle_I2V_LOW.safetensors',
    'NSFW_pack/iGoon - Blink_Missionary_I2V_LOW v2.safetensors',
    'NSFW_pack/iGoon - Blink_Missionary_I2V_LOW.safetensors',
    'NSFW_pack/iGoon - Blink_Back_Doggystyle_LOW.safetensors',
    'NSFW_pack/iGoon - Blink_Facial_I2V_LOW.safetensors',
    'NSFW_pack/iGoon_Blink_Titjob_I2V_LOW.safetensors',
    'NSFW_pack/lips-bj_low_noise.safetensors',
    'NSFW_pack/mql_casting_sex_doggy_kneel_diagonally_behind_vagina_wan22_i2v_v1_low_noise.safetensors',
    'NSFW_pack/mql_casting_sex_reverse_cowgirl_lie_front_vagina_wan22_i2v_v1_low_noise.safetensors',
    'NSFW_pack/mql_casting_sex_spoon_wan22_i2v_v1_low_noise.safetensors',
    'NSFW_pack/mql_doggy_a_wan22_t2v_v1_low_noise.safetensors',
    'NSFW_pack/mql_massage_tits_wan22_i2v_v1_low_noise.safetensors',
    'NSFW_pack/mql_panties_aside_wan22_i2v_v1_low_noise.safetensors',
    'NSFW_pack/mqlspn_a_wan22_t2v_v1_low_noise.safetensors',
    'NSFW_pack/sfbehind_v2.1_low_noise.safetensors',
    'NSFW_pack/sid3l3g_transition_v2.0_L.safetensors',
    'NSFW_pack/wan2.2_i2v_low_ulitmate_pussy_asshole.safetensors',
    // Root loras
    'erect_penis_epoch_80.safetensors',
    'pov-blowjob-i2v-v1.2.safetensors',
    'jfj-deepthroat-W22-I2V-LN.safetensors',
    'front_doggy_plow_v1_1_wan.safetensors',
    'pov-missionary-i2v-low-v1.0.safetensors',
    'side-sex-i2v-v10.safetensors',
    'wan22.r3v3rs3_c0wg1rl-14b-Low-i2v_e70.safetensors',
    'fingering-low-v1.0.safetensors',
    'nipple_stroke_WAN22_I2V_v1_low_noise.safetensors',
];

// Default scene configs
const DEFAULT_DURATIONS = [2.5, 2.5, 2.5, 2.0, 1.5, 4.0];
const DEFAULT_HIGH_LORAS = [
    { on: true, lora: 'HIGH/DR34ML4Y_HIGH_V2.safetensors', strength: 0.9 },
    { on: true, lora: 'HIGH/NSFW-22-H-e8.safetensors', strength: 0.6 },
    { on: false, lora: 'None', strength: 1.0 },
    { on: false, lora: 'None', strength: 1.0 },
];
const DEFAULT_LOW_LORAS = [
    { on: true, lora: 'LOW/DR34ML4Y_LOW_V2.safetensors', strength: 0.8 },
    { on: true, lora: 'LOW/NSFW-22-L-e8.safetensors', strength: 0.6 },
    { on: false, lora: 'None', strength: 1.0 },
    { on: false, lora: 'None', strength: 1.0 },
];

// Scene state (6 scenes)
const kenpechiScenes = [];
for (let i = 0; i < 6; i++) {
    kenpechiScenes.push({
        prompt: '',
        duration: DEFAULT_DURATIONS[i],
        high_loras: JSON.parse(JSON.stringify(DEFAULT_HIGH_LORAS)),
        low_loras: JSON.parse(JSON.stringify(DEFAULT_LOW_LORAS)),
    });
}

let activeLoraScene = 0; // Which scene the LoRA modal is editing

function loraBaseName(path) {
    return path.replace(/^(HIGH|LOW|NSFW_pack)\//, '').replace('.safetensors', '');
}

// Preset prompts — fill all 6 scenes at once
const PRESET_PROMPTS = {
    missionary: `m15510n4ry, a woman is lying on her back with her legs spread looking up at the viewer, having violent sex with a man. Man's big penis immediately thrusting fully deep in and fully out of her vagina, so we can see it, he is piston fucking causing her body hips into a rocking motion while her breasts bounce from each thrust, she bounces forward, her breasts are bouncing. The camera zooms in on the woman's waist. She keeps looking at the camera. Authentic film look,High-fidelity details`,
    blowjob: `bl0wj0b. She sensually starts performing a deepthroat blowjob. She is bobbing her head back and forth slowly while sucking the man's erect penis with the foreskin pulled back, the penis is going deep into her mouth and throat. She rams her head forward, swallowing the entire penis until her nose smashes against his hips, then pulls back gasping for air. The camera zooms in on the man's penis. She keeps looking at the man with eyes open. Authentic film look,High-fidelity details`,
    doggy: `d0gg1e, A woman is having doggy style sex with a man. She thrusts her ass violently towards the camera repeatedly. she is fucking the man by rapidly moving her hips, her buttocks move around. She bounces her ass up and down. jiggle with recoil, rhythmic up-and-down motion with her hips, dynamic hip thrusts, thighs shaking, peak jiggle moments, realistic skin deformation. twerks causing her ass to jiggle and shake. A woman facing forward while turning only her head to look behind her. She stares at the camera with a seductive stare. She keeps looking at the camera. Authentic film look,High-fidelity details`,
    cowgirl: `c0wg1rl,A woman straddling a man who is lying on his back. The woman's legs are spread wide and she is sitting on top of the man in the cowgirl position with his erect penis penetrating her vagina. His penis is going in and out of her pussy. He is piston fucking causing her body hips into a rocking motion while her breasts bounce from each thrust, she bounces forward, her breasts are bouncing. She keeps looking at the camera. Authentic film look, High-fidelity details`,
    handjob: `handj0b. She is gripping his penis with one hand.The mans veiny detailed penis is prominent. During the video she tightens her grip on his penis and quickly strokes the erect firm penis up to the top and down to the bottom giving the man a handjob, stroking his penis quickly and efficiently trying to make the man orgasm as fast as possible. She stares at the camera with a seductive stare. The camera zooms in on his penis. She smiles at camera. She stares at the camera with a seductive stare. She keeps looking at the camera. Authentic film look, High-fidelity details`,
};

function applyPresetPrompt(presetKey) {
    const prompt = PRESET_PROMPTS[presetKey];
    if (!prompt) return;
    for (let i = 0; i < 6; i++) {
        kenpechiScenes[i].prompt = prompt;
    }
    buildSceneCards();
}

// Bind preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const key = btn.dataset.preset;
        applyPresetPrompt(key);
        // Highlight active preset
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

function buildSceneCards() {
    const container = document.getElementById('scenesContainer');
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < 6; i++) {
        const scene = kenpechiScenes[i];
        const card = document.createElement('div');
        card.className = 'scene-card';
        card.innerHTML = `
            <div class="scene-card-header">
                <span class="scene-card-num">SCENE ${i + 1}</span>
                <button class="scene-card-gear" data-scene="${i}" title="LoRA settings">⚙</button>
            </div>
            <textarea rows="2" placeholder="Scene ${i + 1} prompt..." data-scene="${i}" class="scene-prompt-input">${scene.prompt}</textarea>
        `;
        container.appendChild(card);
    }

    // Bind events
    container.querySelectorAll('.scene-prompt-input').forEach(el => {
        el.addEventListener('input', (e) => {
            kenpechiScenes[parseInt(e.target.dataset.scene)].prompt = e.target.value;
        });
    });
    container.querySelectorAll('.scene-card-gear').forEach(el => {
        el.addEventListener('click', (e) => {
            openLoraModal(parseInt(e.currentTarget.dataset.scene));
        });
    });
}

function updateLoraBadges(sceneIdx) {
    const badge = document.getElementById(`loraBadges${sceneIdx}`);
    if (!badge) return;
    const scene = kenpechiScenes[sceneIdx];
    const active = [];
    scene.high_loras.forEach(l => { if (l.on && l.lora !== 'None') active.push(loraBaseName(l.lora)); });
    scene.low_loras.forEach(l => { if (l.on && l.lora !== 'None') active.push(loraBaseName(l.lora)); });
    // Deduplicate
    const unique = [...new Set(active)];
    badge.innerHTML = unique.map(name => `<span class="lora-badge">${name}</span>`).join('');
}

// LoRA Modal
function openLoraModal(sceneIdx) {
    activeLoraScene = sceneIdx;
    const overlay = document.getElementById('loraModalOverlay');
    const title = document.getElementById('loraModalTitle');
    const body = document.getElementById('loraModalBody');
    title.textContent = `Scene ${sceneIdx + 1} — LoRAs`;

    const scene = kenpechiScenes[sceneIdx];
    let html = '<div class="lora-group-title">HIGH Model</div>';
    for (let s = 0; s < 4; s++) {
        const lora = scene.high_loras[s];
        html += buildLoraSlotHTML('high', s, lora, KENPECHI_LORAS_HIGH);
    }
    html += '<div class="lora-group-title">LOW Model</div>';
    for (let s = 0; s < 4; s++) {
        const lora = scene.low_loras[s];
        html += buildLoraSlotHTML('low', s, lora, KENPECHI_LORAS_LOW);
    }
    body.innerHTML = html;

    overlay.style.display = 'flex';
}

function buildLoraSlotHTML(type, slot, lora, loraList) {
    const selId = `lora_${type}_${slot}`;
    const strId = `str_${type}_${slot}`;
    let options = `<option value="None"${!lora.on || lora.lora === 'None' ? ' selected' : ''}>Off</option>`;
    loraList.forEach(name => {
        const sel = lora.on && lora.lora === name ? ' selected' : '';
        options += `<option value="${name}"${sel}>${loraBaseName(name)}</option>`;
    });
    return `
        <div class="lora-slot">
            <select id="${selId}">${options}</select>
            <input type="number" class="lora-str-input" id="${strId}" min="0.1" max="2.0" step="0.1" value="${lora.strength.toFixed(1)}">
        </div>
    `;
}

function closeLoraModal() {
    // Save values back to state
    const scene = kenpechiScenes[activeLoraScene];
    for (let s = 0; s < 4; s++) {
        const highSel = document.getElementById(`lora_high_${s}`);
        const highStr = document.getElementById(`str_high_${s}`);
        if (highSel) {
            scene.high_loras[s] = {
                on: highSel.value !== 'None',
                lora: highSel.value === 'None' ? 'None' : highSel.value,
                strength: parseFloat(highStr?.value || 1.0),
            };
        }
        const lowSel = document.getElementById(`lora_low_${s}`);
        const lowStr = document.getElementById(`str_low_${s}`);
        if (lowSel) {
            scene.low_loras[s] = {
                on: lowSel.value !== 'None',
                lora: lowSel.value === 'None' ? 'None' : lowSel.value,
                strength: parseFloat(lowStr?.value || 1.0),
            };
        }
    }
    updateLoraBadges(activeLoraScene);
    document.getElementById('loraModalOverlay').style.display = 'none';
}

// Bind modal close & save
document.getElementById('loraModalClose')?.addEventListener('click', closeLoraModal);
document.getElementById('loraModalSave')?.addEventListener('click', closeLoraModal);
document.getElementById('loraModalOverlay')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeLoraModal();
});

// Advanced settings toggle
// Build scene cards on page load
buildSceneCards();

// Override generateVideo for Kenpechi
const _originalGenerateVideo = typeof generateVideo === 'function' ? generateVideo : null;

async function generateKenpechiVideo() {
    if (!originalImage) {
        alert('Please upload a photo first');
        return;
    }

    // Collect scenes (only non-empty prompts)
    const scenes = kenpechiScenes
        .filter(s => s.prompt.trim())
        .map(s => ({
            prompt: s.prompt,
            duration: s.duration,
            seed: -1,
            high_loras: s.high_loras,
            low_loras: s.low_loras,
        }));

    if (scenes.length === 0) {
        alert('Please add a prompt to at least one scene');
        return;
    }

    // Pad to 6 scenes if less (reuse last prompt)
    while (scenes.length < 6) {
        scenes.push({ ...scenes[scenes.length - 1], seed: -1 });
    }

    const generateBtn = document.getElementById('generateBtn');
    const progressInfo = document.getElementById('progressInfo');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultSection = document.getElementById('resultSection');

    // UI state
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').style.display = 'none';
    generateBtn.querySelector('.btn-loader').style.display = '';
    progressInfo.style.display = '';
    resultSection.style.display = 'none';
    progressFill.style.width = '5%';
    progressText.textContent = 'Submitting Kenpechi job...';

    const cancelBtn = document.getElementById('cancelVideoBtn');
    if (cancelBtn) cancelBtn.style.display = '';

    try {
        const imageDataURL = getImageDataURL();
        const imageB64 = imageDataURL.split(',')[1];

        const body = {
            image: imageB64,
            scenes: scenes,
            negative: document.getElementById('kenpechiNegativeInput')?.value || '',
            width: 720,
            height: 1072,
            steps: 7,
            split_steps: 3,
            fps: 16,
            rife_multiplier: 2,
            svi_motion_strength: 1.0,
            repulsion_boost: 1.0,
            shift: 5.0,
        };

        const submitResp = await fetch('/api/video/kenpechi', {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(body),
        });

        if (!submitResp.ok) {
            const errData = await submitResp.json().catch(() => null);
            throw new Error(errData?.error || `Server error: ${submitResp.status}`);
        }

        const submitData = await submitResp.json();
        if (!submitData.job_id) throw new Error('No job_id in response');

        currentVideoJobId = submitData.job_id;
        console.log('Kenpechi job submitted:', currentVideoJobId);
        progressFill.style.width = '10%';
        progressText.textContent = 'Job queued...';

        // Poll for status
        let pollCount = 0;
        const maxPolls = 600; // 30min max
        while (pollCount < maxPolls) {
            await new Promise(r => setTimeout(r, 3000));
            pollCount++;

            try {
                const statusResp = await fetch(`/api/video/status/${currentVideoJobId}`, {
                    headers: authHeaders(),
                });
                if (!statusResp.ok) continue;
                const statusData = await statusResp.json();

                if (statusData.status === 'COMPLETED') {
                    progressFill.style.width = '100%';
                    progressText.textContent = 'Done!';

                    if (statusData.video) {
                        const resultVideo = document.getElementById('resultVideo');
                        const resultImage = document.getElementById('resultImage');
                        lastResultB64 = statusData.video;
                        lastResultType = 'video';
                        resultVideo.src = 'data:video/mp4;base64,' + statusData.video;
                        resultVideo.style.display = '';
                        resultImage.style.display = 'none';
                        resultSection.style.display = '';
                        resultSection.scrollIntoView({ behavior: 'smooth' });
                    }
                    break;
                } else if (statusData.status === 'FAILED') {
                    throw new Error(statusData.error || 'Job failed');
                } else {
                    const pct = Math.min(10 + pollCount * 0.5, 95);
                    progressFill.style.width = pct + '%';
                    progressText.textContent = statusData.status === 'IN_PROGRESS' ? 'Generating...' : 'Waiting for GPU...';
                }
            } catch (pollErr) {
                console.warn('Poll error:', pollErr);
            }
        }

    } catch (err) {
        progressFill.style.width = '0%';
        progressText.textContent = 'Error: ' + err.message;
        alert('Error: ' + err.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').style.display = '';
        generateBtn.querySelector('.btn-loader').style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';
    }
}

// Monkey-patch: if mode is 'video', use Kenpechi instead of old generateVideo
const _origPrepareAndGenerate = typeof prepareAndGenerate === 'function' ? prepareAndGenerate : null;
