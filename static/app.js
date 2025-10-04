const form = document.getElementById('uploadForm');
const statusDiv = document.getElementById('status');
const summaryDiv = document.getElementById('summary');
const jsonPre = document.getElementById('json');

const videoInput = document.getElementById('videoInput');
const hiddenVideo = document.getElementById('hiddenVideo');
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

const coordPreview = document.getElementById('coordPreview');
const btnReset = document.getElementById('btnReset');
const btnFlip = document.getElementById('btnFlip');

const x1El = document.getElementById('x1');
const y1El = document.getElementById('y1');
const x2El = document.getElementById('x2');
const y2El = document.getElementById('y2');

let hasFrame = false;
let drawing = false;
let pA = null; // {x,y} em pixels de canvas
let pB = null;

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawFrameToCanvas() {
  const vw = hiddenVideo.videoWidth;
  const vh = hiddenVideo.videoHeight;
  if (!vw || !vh) return;

  // Redimensiona o canvas para manter a proporção do vídeo dentro do tamanho atual do canvas
  // Aqui vamos preencher o canvas mantendo proporção (letterbox se necessário)
  const targetW = canvas.clientWidth || canvas.width;
  const aspect = vw / vh;
  // Ajusta tamanho real (atributos width/height) para evitar blur
  canvas.width = Math.round(targetW);
  canvas.height = Math.round(targetW / aspect);

  ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);
  hasFrame = true;
  redrawLine();
}

function redrawLine() {
  if (!hasFrame) return;
  // redesenha o frame atual do vídeo como fundo
  ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);

  if (pA && pB) {
    // linha
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#FFD400';
    ctx.beginPath();
    ctx.moveTo(pA.x, pA.y);
    ctx.lineTo(pB.x, pB.y);
    ctx.stroke();

    // pontos
    ctx.fillStyle = '#0d6efd';
    ctx.beginPath(); ctx.arc(pA.x, pA.y, 6, 0, Math.PI*2); ctx.fill();

    ctx.fillStyle = '#dc3545';
    ctx.beginPath(); ctx.arc(pB.x, pB.y, 6, 0, Math.PI*2); ctx.fill();
  }

  updateHiddenInputs();
}

function updateHiddenInputs() {
  if (!pA || !pB || !hasFrame) {
    x1El.value = ''; y1El.value = ''; x2El.value = ''; y2El.value = '';
    coordPreview.textContent = 'x1=— y1=— | x2=— y2=—';
    return;
  }
  // normaliza 0..1 relativamente ao tamanho atual do canvas
  const x1n = (pA.x / canvas.width);
  const y1n = (pA.y / canvas.height);
  const x2n = (pB.x / canvas.width);
  const y2n = (pB.y / canvas.height);

  x1El.value = x1n.toFixed(4);
  y1El.value = y1n.toFixed(4);
  x2El.value = x2n.toFixed(4);
  y2El.value = y2n.toFixed(4);

  coordPreview.textContent = `x1=${x1El.value} y1=${y1El.value} | x2=${x2El.value} y2=${y2El.value}`;
}

function setLine(a, b) {
  pA = a ? {x: a.x, y: a.y} : null;
  pB = b ? {x: b.x, y: b.y} : null;
  redrawLine();
}

function flipLine() {
  if (pA && pB) {
    const tmp = pA; pA = pB; pB = tmp;
    redrawLine();
  }
}

btnReset.addEventListener('click', () => setLine(null, null));
btnFlip.addEventListener('click', flipLine);

// Eventos de desenho
canvas.addEventListener('mousedown', (e) => {
  if (!hasFrame) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  drawing = true;
  setLine({x, y}, null);
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing || !hasFrame) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  pB = {x, y};
  redrawLine();
});

window.addEventListener('mouseup', () => { drawing = false; });

// Suporte básico a touch
canvas.addEventListener('touchstart', (e) => {
  if (!hasFrame) return;
  const t = e.touches[0];
  const r = canvas.getBoundingClientRect();
  setLine({x: t.clientX - r.left, y: t.clientY - r.top}, null);
  e.preventDefault();
});
canvas.addEventListener('touchmove', (e) => {
  if (!hasFrame || !pA) return;
  const t = e.touches[0];
  const r = canvas.getBoundingClientRect();
  pB = {x: t.clientX - r.left, y: t.clientY - r.top};
  redrawLine();
  e.preventDefault();
});
canvas.addEventListener('touchend', () => { /* nada */ });

// Carrega primeiro frame do vídeo selecionado
videoInput.addEventListener('change', async () => {
  const file = videoInput.files?.[0];
  if (!file) return;
  setLine(null, null);

  const url = URL.createObjectURL(file);
  hiddenVideo.src = url;
  hiddenVideo.currentTime = 0;

  // aguarda metadados para saber dimensões
  await hiddenVideo.play().catch(()=>{});
  hiddenVideo.pause();

  // tenta capturar o frame em ~0.1s para evitar quadros pretos
  hiddenVideo.currentTime = 0.1;
  hiddenVideo.addEventListener('seeked', onSeekedOnce, { once: true });
});

function onSeekedOnce() {
  drawFrameToCanvas();
}

// Redesenha ao redimensionar janela (recalcula canvas e re-exibe frame/linha)
window.addEventListener('resize', () => {
  if (hasFrame) drawFrameToCanvas();
});

// Submissão do formulário
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  statusDiv.textContent = 'Processando (stream) …';
  summaryDiv.innerHTML = '';
  jsonPre.textContent = '';

  const fd = new FormData(form);

  // Use /upload_stream para receber eventos parciais
  const resp = await fetch('/upload_stream', { method: 'POST', body: fd });
  if (!resp.ok || !resp.body) {
    statusDiv.textContent = 'Erro ❌';
    jsonPre.textContent = 'Falha ao iniciar streaming';
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let totalsIn = 0, totalsOut = 0;
  const chunkRows = []; // para tabela de chunks
  let finalData = null;

  function renderPartial() {
    // desenha um resumo parcial simples dos chunks processados
    let html = `<div class="pill">IN parcial: <strong>${totalsIn}</strong></div>
                <div class="pill">OUT parcial: <strong>${totalsOut}</strong></div>`;
    if (chunkRows.length) {
      html += `<table border="1" cellpadding="6" cellspacing="0" style="margin-top:8px;">
        <tr><th>Chunk</th><th>Janela</th><th>IN</th><th>OUT</th></tr>
        ${chunkRows.map(r => `<tr><td>${r.index}</td><td>${r.label}</td><td>${r.in}</td><td>${r.out}</td></tr>`).join('')}
      </table>`;
    }
    summaryDiv.innerHTML = html;
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // processa por linhas
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;

      let ev;
      try { ev = JSON.parse(line); } catch { continue; }

      if (ev.event === 'start') {
        statusDiv.textContent = 'Processando (chunks chegando)…';
      } else if (ev.event === 'chunk_result') {
        // acumula parciais
        totalsIn += ev.in || 0;
        totalsOut += ev.out || 0;
        const toMMSS = (sec) => {
          const m = Math.floor(sec/60), s = Math.floor(sec%60);
          return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
        };
        chunkRows.push({
          index: ev.index,
          label: `${toMMSS(ev.startSec)}-${toMMSS(ev.endSec)}`,
          in: ev.in, out: ev.out
        });
        renderPartial();
      } else if (ev.event === 'final') {
        finalData = ev;
      } else if (ev.event === 'error') {
        statusDiv.textContent = 'Erro ❌';
        jsonPre.textContent = ev.detail || 'Erro no streaming';
      }
    }
  }

  // render final
  if (finalData) {
    statusDiv.textContent = 'Concluído ✅';
    const s = finalData.totals || {};
    const linkVid = finalData.annotated_video_url ? `<a class="pill" href="${finalData.annotated_video_url}" target="_blank">Baixar vídeo anotado</a>` : '';

    let windowsHtml = '';
    if (finalData.windows && finalData.windows.length) {
      windowsHtml = `<h4>Faixas de horário</h4>
      <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Janela</th><th>Entraram (IN)</th><th>Sairam (OUT)</th></tr>
        ${finalData.windows.map(w => `<tr><td>${w.label}</td><td>${w.in}</td><td>${w.out}</td></tr>`).join('')}
      </table>`;
    }

    summaryDiv.innerHTML = `
      ${linkVid}
      <div class="pill">IN total: <strong>${s.in ?? 0}</strong></div>
      <div class="pill">OUT total: <strong>${s.out ?? 0}</strong></div>
      <p><strong>Linha (px):</strong> (${finalData.linePx?.x1},${finalData.linePx?.y1}) → (${finalData.linePx?.x2},${finalData.linePx?.y2})</p>
      ${windowsHtml}
    `;

    // joga a timeline (primeiras 80 linhas) no JSON
    jsonPre.textContent = JSON.stringify(finalData.timeline?.slice(0, 80) || [], null, 2)
      + ((finalData.timeline?.length || 0) > 80 ? `\n... (+${finalData.timeline.length - 80} linhas)` : '');
  }
});