/* ══════════════════════════════════════════
   ChurnIQ — Frontend Application Logic
══════════════════════════════════════════ */

let lastResult = null;

// ─── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  initSegControls();
  initToggleGroups();
  checkApiHealth();
});

// ─── SLIDER INIT ──────────────────────────────────────────────────────────
function initSliders() {
  const tenure = document.getElementById('tenure');
  const tenureVal = document.getElementById('tenure-val');
  const charges = document.getElementById('monthly-charges');
  const chargesVal = document.getElementById('charges-val');

  tenure.addEventListener('input', () => {
    tenureVal.textContent = `${tenure.value} mo`;
    updateSliderFill(tenure);
  });

  charges.addEventListener('input', () => {
    chargesVal.textContent = `$${charges.value}`;
    updateSliderFill(charges);
  });

  updateSliderFill(tenure);
  updateSliderFill(charges);
}

function updateSliderFill(slider) {
  const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
  slider.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pct}%, #1e293b ${pct}%)`;
}

// ─── SEGMENTED CONTROLS ───────────────────────────────────────────────────
function initSegControls() {
  document.querySelectorAll('.seg-control').forEach(ctrl => {
    const hiddenId = ctrl.id.replace('-seg', '');
    const hidden = document.getElementById(hiddenId);
    ctrl.querySelectorAll('.seg').forEach(btn => {
      btn.addEventListener('click', () => {
        ctrl.querySelectorAll('.seg').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (hidden) hidden.value = btn.dataset.val;
      });
    });
  });
}

// ─── TOGGLE GROUPS ────────────────────────────────────────────────────────
function initToggleGroups() {
  const toggleMap = {
    'security-toggle': 'online-security',
    'support-toggle':  'tech-support',
    'senior-toggle':   'senior',
    'tv-toggle':       'streaming-tv',
    'movies-toggle':   'streaming-movies',
  };

  Object.entries(toggleMap).forEach(([groupId, hiddenId]) => {
    const group  = document.getElementById(groupId);
    const hidden = document.getElementById(hiddenId);
    if (!group) return;
    group.querySelectorAll('.tog').forEach(btn => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.tog').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        hidden.value = btn.dataset.val;
      });
    });
  });
}

// ─── COLLECT INPUTS ───────────────────────────────────────────────────────
function collectInputs() {
  const tenure = parseInt(document.getElementById('tenure').value);
  const monthly = parseFloat(document.getElementById('monthly-charges').value);
  return {
    gender:          'Male',
    SeniorCitizen:   parseInt(document.getElementById('senior').value),
    Partner:         'No',
    Dependents:      'No',
    tenure:          tenure,
    PhoneService:    'Yes',
    MultipleLines:   'No',
    InternetService: document.getElementById('internet-service').value,
    OnlineSecurity:  document.getElementById('online-security').value,
    OnlineBackup:    'No',
    DeviceProtection:'No',
    TechSupport:     document.getElementById('tech-support').value,
    StreamingTV:     document.getElementById('streaming-tv').value,
    StreamingMovies: document.getElementById('streaming-movies').value,
    Contract:        document.getElementById('contract').value,
    PaperlessBilling:'Yes',
    PaymentMethod:   document.getElementById('payment-method').value,
    MonthlyCharges:  monthly,
    TotalCharges:    tenure * monthly,
  };
}

// ─── PREDICT ──────────────────────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.querySelector('.btn-text').textContent = 'Analyzing…';
  btn.querySelector('.btn-icon').innerHTML = '<span class="spinner"></span>';

  try {
    const payload = collectInputs();
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`API error ${res.status}`);
    const data = await res.json();
    lastResult = { ...data, inputs: payload };
    renderResults(data);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = 'Analyze Customer Risk';
    btn.querySelector('.btn-icon').textContent = '→';
  }
}

// ─── RENDER RESULTS ───────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById('placeholder').classList.add('hidden');
  const content = document.getElementById('results-content');
  content.classList.remove('hidden');

  // Gauge
  drawGauge(data.churn_probability);
  animateNumber(document.getElementById('gauge-prob'), 0, Math.round(data.churn_probability * 100), '%');

  // Risk badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = `${data.risk_level.toUpperCase()} RISK`;
  badge.className = `risk-badge ${data.risk_level.toLowerCase()}`;

  // Stats
  animateNumber(document.getElementById('stat-annual'), 0, data.annual_revenue_at_risk, '', '$');
  animateNumber(document.getElementById('stat-clv'),    0, data.clv_estimate,           '', '$');
  document.getElementById('stat-actions').textContent = data.recommendations.length;

  // Recommendations
  const list = document.getElementById('recs-list');
  list.innerHTML = '';
  data.recommendations.forEach((rec, i) => {
    const card = document.createElement('div');
    card.className = 'rec-card';
    card.style.animationDelay = `${i * 80}ms`;
    card.innerHTML = `
      <div class="rec-icon">${rec.icon}</div>
      <div class="rec-body">
        <div class="rec-title">${rec.title}</div>
        <div class="rec-detail">${rec.detail}</div>
      </div>
      <div class="rec-impact impact-${rec.impact}">${rec.impact}</div>
    `;
    list.appendChild(card);
  });

  content.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── GAUGE CANVAS ─────────────────────────────────────────────────────────
function drawGauge(probability) {
  const canvas = document.getElementById('gauge-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 20;
  const r  = Math.min(W, H * 1.6) / 2 - 20;

  const startA = Math.PI;
  const endA   = 2 * Math.PI;

  // Track bg
  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, endA);
  ctx.lineWidth  = 18;
  ctx.strokeStyle = '#1e293b';
  ctx.lineCap    = 'round';
  ctx.stroke();

  // Colored arc
  const fillA = startA + probability * Math.PI;
  const color  = probability > 0.6 ? '#f87171' : probability > 0.4 ? '#fbbf24' : '#34d399';
  const grd    = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grd.addColorStop(0,   '#34d399');
  grd.addColorStop(0.4, '#fbbf24');
  grd.addColorStop(1,   '#f87171');

  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, fillA);
  ctx.lineWidth   = 18;
  ctx.strokeStyle = grd;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Needle
  const needleA = startA + probability * Math.PI;
  const nx = cx + (r - 10) * Math.cos(needleA);
  const ny = cy + (r - 10) * Math.sin(needleA);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.lineWidth   = 2;
  ctx.strokeStyle = '#fff';
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();

  // Labels
  ctx.fillStyle   = '#64748b';
  ctx.font        = 'bold 11px DM Sans, sans-serif';
  ctx.textAlign   = 'left';
  ctx.fillText('0%', cx - r - 4, cy + 18);
  ctx.textAlign   = 'right';
  ctx.fillText('100%', cx + r + 4, cy + 18);
}

// ─── ANIMATE NUMBER ───────────────────────────────────────────────────────
function animateNumber(el, from, to, suffix = '', prefix = '') {
  const dur  = 800;
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val  = from + (to - from) * ease;
    el.textContent = prefix + (Number.isInteger(to) ? Math.round(val) : val.toFixed(0)) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─── DOWNLOAD REPORT ──────────────────────────────────────────────────────
function downloadReport() {
  if (!lastResult) return;
  const d = lastResult;
  const inp = d.inputs;
  const recs = d.recommendations.map(r => `  [${r.impact} Impact] ${r.title}: ${r.detail}`).join('\n');
  const ts   = new Date().toLocaleString();

  const txt = `
╔══════════════════════════════════════════════════════════╗
║          ChurnIQ — Customer Retention Report              ║
╚══════════════════════════════════════════════════════════╝

Generated: ${ts}

┌─ RISK ASSESSMENT ─────────────────────────────────────────
│  Churn Probability    : ${(d.churn_probability * 100).toFixed(1)}%
│  Risk Level           : ${d.risk_level}
│  Annual Revenue Risk  : $${d.annual_revenue_at_risk.toFixed(2)}
│  Customer CLV Est.    : $${d.clv_estimate.toFixed(2)}
└───────────────────────────────────────────────────────────

┌─ CUSTOMER PROFILE ────────────────────────────────────────
│  Tenure               : ${inp.tenure} months
│  Monthly Charges      : $${inp.MonthlyCharges.toFixed(2)}
│  Total Charges        : $${inp.TotalCharges.toFixed(2)}
│  Contract Type        : ${inp.Contract}
│  Internet Service     : ${inp.InternetService}
│  Online Security      : ${inp.OnlineSecurity}
│  Tech Support         : ${inp.TechSupport}
│  Payment Method       : ${inp.PaymentMethod}
│  Streaming TV         : ${inp.StreamingTV}
│  Streaming Movies     : ${inp.StreamingMovies}
└───────────────────────────────────────────────────────────

┌─ STRATEGIC RETENTION ACTIONS ─────────────────────────────
${recs}
└───────────────────────────────────────────────────────────

Powered by ChurnIQ v2.0 — FastAPI + ML Pipeline
`;

  const blob = new Blob([txt.trim()], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `churniq_report_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── SCROLL HELPER ────────────────────────────────────────────────────────
function scrollToSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
}

// ─── API HEALTH CHECK ─────────────────────────────────────────────────────
async function checkApiHealth() {
  const el = document.getElementById('api-status');
  try {
    const [healthRes, statsRes] = await Promise.all([
      fetch('/health'),
      fetch('/api/stats')
    ]);
    const health = await healthRes.json();
    const stats  = await statsRes.json();
    if (health.status === 'ok') {
      el.textContent = `API: Online · ${stats.model || 'Model Ready'}`;
      el.style.color = 'var(--safe)';
    }
  } catch {
    el.textContent = 'API: Check server is running';
    el.style.color = 'var(--warn)';
  }
}
