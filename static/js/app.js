/* ====================================================================
   DeepFake Detector — Frontend Logic
   ==================================================================== */

let selectedImageFile = null;
let selectedVideoFile = null;

// ── Tab switching ─────────────────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
}

// ── Drag-and-drop support ─────────────────────────────────────────────

function setupDragDrop(zoneId, inputId, handler) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handler({ target: { files: [file] } });
  });
  zone.addEventListener('click', () => document.getElementById(inputId).click());
}

document.addEventListener('DOMContentLoaded', () => {
  setupDragDrop('img-drop-zone', 'img-input', handleImageSelect);
  setupDragDrop('vid-drop-zone', 'vid-input', handleVideoSelect);
});

// ── Image flow ────────────────────────────────────────────────────────

function handleImageSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Please upload a JPG, PNG, or WEBP image.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Maximum size is 10 MB.', 'error');
    return;
  }

  selectedImageFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('img-preview').src = e.target.result;
    document.getElementById('img-drop-zone').classList.add('hidden');
    document.getElementById('img-preview-wrapper').classList.remove('hidden');
    document.getElementById('img-result').classList.add('hidden');
    document.getElementById('heatmap-card').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

async function detectImage() {
  if (!selectedImageFile) return;

  const btn = document.getElementById('img-detect-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analysing…';

  const formData = new FormData();
  formData.append('file', selectedImageFile);

  try {
    const resp = await fetch('/predict/image', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      showToast(data.error, 'error');
      return;
    }

    renderImageResult(data);

  } catch (err) {
    showToast('Network error. Is the server running?', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyse Image';
  }
}

function renderImageResult(data) {
  const box = document.getElementById('img-result');
  const label = data.label;
  const conf  = data.confidence;

  if (data.heatmap_url) {
    document.getElementById('heatmap-preview').src = data.heatmap_url + '?t=' + Date.now();
    document.getElementById('heatmap-card').style.display = 'block';
  }

  box.innerHTML = `
    <div class="result-header">
      <div class="verdict ${label}">${label === 'REAL' ? '✅' : '🚨'} ${label}</div>
    </div>
    <div class="confidence-label">Confidence: <strong>${conf}%</strong></div>
    <div class="confidence-bar-bg">
      <div class="confidence-bar-fill ${label}" style="width: ${conf}%"></div>
    </div>
    <div class="result-meta">
      <span class="meta-pill">Model: EfficientNet-B4</span>
      <span class="meta-pill">${data.face_detected ? '✓ Face detected' : '⚠ No face — full image used'}</span>
      <span class="meta-pill">Raw score: ${data.details?.raw_score ?? 'N/A'}</span>
    </div>
    ${!data.face_detected ? '<p style="color:var(--warn);font-size:0.85rem;margin-top:10px;">⚠ No face was detected. Results may be less accurate.</p>' : ''}
  `;
  box.classList.remove('hidden');
}

function resetImage() {
  selectedImageFile = null;
  document.getElementById('img-input').value = '';
  document.getElementById('img-drop-zone').classList.remove('hidden');
  document.getElementById('img-preview-wrapper').classList.add('hidden');
  document.getElementById('img-result').classList.add('hidden');
  document.getElementById('heatmap-card').style.display = 'none';
}

// ── Video flow ────────────────────────────────────────────────────────

function handleVideoSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
  if (!allowed.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
    showToast('Please upload a MP4, AVI, or MOV video.', 'error');
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    showToast('File too large. Maximum size is 100 MB.', 'error');
    return;
  }

  selectedVideoFile = file;
  const url = URL.createObjectURL(file);
  document.getElementById('vid-preview').src = url;
  document.getElementById('vid-drop-zone').classList.add('hidden');
  document.getElementById('vid-preview-wrapper').classList.remove('hidden');
  document.getElementById('vid-result').classList.add('hidden');
}

async function detectVideo() {
  if (!selectedVideoFile) return;

  const btn = document.getElementById('vid-detect-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Extracting frames & analysing…';

  const formData = new FormData();
  formData.append('file', selectedVideoFile);

  try {
    const resp = await fetch('/predict/video', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      showToast(data.error, 'error');
      return;
    }

    renderVideoResult(data);

  } catch (err) {
    showToast('Network error. Is the server running?', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyse Video';
  }
}

function renderVideoResult(data) {
  const box   = document.getElementById('vid-result');
  const label = data.label;
  const conf  = data.confidence;

  let frameChart = '';
  if (data.per_frame_scores && data.per_frame_scores.length > 0) {
    const bars = data.per_frame_scores.map(score => {
      const isFake = score > 50;
      const h = Math.max(4, Math.round(score / 2));
      return `<div class="frame-bar" style="height:${h}px;background:${isFake ? 'var(--fake)' : 'var(--real)'};opacity:0.85;" title="${score}% fake"></div>`;
    }).join('');
    frameChart = `
      <div class="frame-chart">
        <p class="frame-chart-title">Per-frame fake probability (hover for value):</p>
        <div class="frame-bars">${bars}</div>
      </div>
    `;
  }

  box.innerHTML = `
    <div class="result-header">
      <div class="verdict ${label}">${label === 'REAL' ? '✅' : '🚨'} ${label}</div>
    </div>
    <div class="confidence-label">Average confidence: <strong>${conf}%</strong></div>
    <div class="confidence-bar-bg">
      <div class="confidence-bar-fill ${label}" style="width: ${conf}%"></div>
    </div>
    <div class="result-meta">
      <span class="meta-pill">Frames analysed: ${data.frames_analyzed}</span>
      <span class="meta-pill">Frames with faces: ${data.frames_with_faces}</span>
      <span class="meta-pill">Model: EfficientNet-B4</span>
    </div>
    ${frameChart}
    ${!data.face_detected ? '<p style="color:var(--warn);font-size:0.85rem;margin-top:10px;">⚠ No faces detected in any frame.</p>' : ''}
  `;
  box.classList.remove('hidden');
}

function resetVideo() {
  selectedVideoFile = null;
  document.getElementById('vid-input').value = '';
  document.getElementById('vid-drop-zone').classList.remove('hidden');
  document.getElementById('vid-preview-wrapper').classList.add('hidden');
  document.getElementById('vid-result').classList.add('hidden');
}

// ── Toast notification ────────────────────────────────────────────────

function showToast(message, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'toast';
  toast.style.cssText = `
    position: fixed; bottom: 28px; right: 28px; z-index: 9999;
    background: ${type === 'error' ? '#ef4444' : '#6c63ff'};
    color: white; padding: 12px 20px; border-radius: 10px;
    font-size: 0.9rem; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease;
    max-width: 320px; line-height: 1.4;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 4000);
}

const style = document.createElement('style');
style.textContent = `@keyframes slideIn { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }`;
document.head.appendChild(style);
