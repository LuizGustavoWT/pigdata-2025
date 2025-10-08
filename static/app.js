// static/app.js — BusSight • Front-end (upload, linha no canvas, processamento, SSE, calculadora)
// Versão PRO: deltas visuais, taxa/min, reconexão SSE, DPR canvas, tolerância visual, atalhos e persistência

(() => {
  const el = (id) => document.getElementById(id);

  const videoInput = el('video');
  const video = el('videoPreview');
  const canvas = el('overlay');
  const ctx = canvas.getContext('2d');

  const x1I = el('x1'), y1I = el('y1'), x2I = el('x2'), y2I = el('y2');
  const sampleFpsI = el('sample_fps'), chunkI = el('chunk_seconds'), workersI = el('workers'), saveAnnI = el('save_annotated');
  const btnProcess = el('btnProcess'), btnStream = el('btnStream');
  const statusEl = el('status');
  const btnReset = el('btnReset'); // (opcional) adicione <button id="btnReset" ...>Reset</button>

  const inCount = el('inCount'), outCount = el('outCount'), netCount = el('netCount');
  const downloads = el('downloads');
  const windowsTable = el('windowsTable');

  // tolerância visual (px na tela). Se não existir input, usa 8.
  const toleranceI = el('tolerance');
  const getTolerancePx = () => {
    const v = parseFloat(toleranceI?.value);
    return Number.isFinite(v) ? Math.max(0, v) : 8;
  };

  let currentVideoPath = null;

  // --- Controle do SSE / auto-stream ---
  let currentEventSource = null;
  let autoStreaming = false;
  let rearmTimer = null;
  let sseRetry = { tries: 0, max: 5 }; // reconexão

  // Linha normalizada [0..1]
  let line = loadLine() ?? { x1: 0.2, y1: 0.8, x2: 0.8, y2: 0.8 };
  let dragMode = null; // 'p1' | 'p2' | 'line'
  let dragOffset = {dx:0, dy:0};
  let hover = false;

  // Contagem incremental (UI)
  let lastPartial = { in: 0, out: 0 };
  let firstProgressTs = null;
  let lastProgressTs = null;

  // ===== Utils/UI =====
  function toast(msg, type='info') {
    const box = document.getElementById('toasts') || document.body;
    const div = document.createElement('div');
    div.className = `toast ${type}`;
    div.textContent = msg;
    box.appendChild(div);
    setTimeout(() => div.remove(), 3200);
  }

  function lockUI(locked) {
    [btnProcess, btnStream, videoInput, x1I,y1I,x2I,y2I,sampleFpsI,chunkI,workersI,saveAnnI,toleranceI,btnReset]
      .filter(Boolean).forEach(b => b.disabled = !!locked);
  }

  const setStatus = throttle((txt, pct=null) => {
    statusEl.textContent = txt;
    if (pct !== null) statusEl.setAttribute('aria-valuenow', pct);
  }, 120);

  function formatBRL(n){ 
    if (!isFinite(n)) n = 0;
    return n.toLocaleString('pt-BR',{style:'currency',currency:'BRL'});
  }
  const clamp01 = (v)=> Math.min(1, Math.max(0, (isFinite(v)?v:0)));

  // ===== Persistência =====
  function saveLine(){ try { localStorage.setItem('bus_line', JSON.stringify(line)); } catch(_){} }
  function loadLine(){
    try { const j = localStorage.getItem('bus_line'); return j? JSON.parse(j) : null; } catch(_){ return null; }
  }
  function saveParams(){
    try {
      const obj = {
        sample_fps: sampleFpsI?.value, chunk_seconds: chunkI?.value, workers: workersI?.value,
        tolerance: toleranceI?.value
      };
      localStorage.setItem('bus_params', JSON.stringify(obj));
    } catch(_){}
  }
  function loadParams(){
    try {
      const j = localStorage.getItem('bus_params'); if (!j) return;
      const p = JSON.parse(j);
      if (sampleFpsI && p.sample_fps) sampleFpsI.value = p.sample_fps;
      if (chunkI && p.chunk_seconds) chunkI.value = p.chunk_seconds;
      if (workersI && p.workers) workersI.value = p.workers;
      if (toleranceI && p.tolerance) toleranceI.value = p.tolerance;
    } catch(_){}
  }
  loadParams();

  // ===== Upload + preview (auto-stream) =====
  async function uploadVideo(file) {
    const form = new FormData();
    form.append('video', file);
    const res = await safeFetch('/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(res.error || 'Falha no upload.');
    currentVideoPath = res.video_path;

    const url = URL.createObjectURL(file);
    video.src = url;
    await video.play().catch(()=>{});
    syncCanvasSize();
    drawLine(true);
    toast('Vídeo carregado. Iniciando contagem…', 'ok');

    // reset contadores locais
    lastPartial = { in: 0, out: 0 };
    firstProgressTs = null;
    lastProgressTs = null;
    flashCounts(0,0,true);

    autoStreaming = true;
    stopStream();
    processStream();
  }

  const drop = document.getElementById('dropArea');
  drop?.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.classList.add('on'); });
  drop?.addEventListener('dragleave', ()=> drop.classList.remove('on'));
  drop?.addEventListener('drop', (e)=>{
    e.preventDefault();
    drop.classList.remove('on');
    const f = e.dataTransfer.files?.[0];
    if (f) uploadVideo(f).catch(err => toast(err.message, 'err'));
  });
  videoInput?.addEventListener('change', (e)=>{
    const f = e.target.files?.[0];
    if (f) uploadVideo(f).catch(err => toast(err.message, 'err'));
  });

  // ===== Canvas DPR & Resize =====
  function syncCanvasSize() {
    const r = video.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
    canvas.width = Math.floor((video.videoWidth || r.width) * dpr);
    canvas.height = Math.floor((video.videoHeight || r.height) * dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  window.addEventListener('resize', ()=>{ syncCanvasSize(); drawLine(); });
  video.addEventListener('loadedmetadata', ()=>{ syncCanvasSize(); drawLine(true); });
  video.addEventListener('play', ()=>{ requestAnimationFrame(tick); });

  function tick() {
    drawLine();
    if (!video.paused && !video.ended) requestAnimationFrame(tick);
  }

  // ===== Helpers normalização =====
  const toPxX = (nx)=> nx * parseFloat(canvas.style.width || canvas.width);
  const toPxY = (ny)=> ny * parseFloat(canvas.style.height || canvas.height);
  const toNrmX = (px)=> clamp01(px / parseFloat(canvas.style.width || canvas.width));
  const toNrmY = (py)=> clamp01(py / parseFloat(canvas.style.height || canvas.height));

  // ===== Desenho da linha + banda de tolerância =====
  function drawLine(forceHandles=false) {
    const cw = parseFloat(canvas.style.width || canvas.width);
    const ch = parseFloat(canvas.style.height || canvas.height);
    ctx.clearRect(0,0,cw,ch);

    // banda (tolerância visual)
    const t = getTolerancePx();
    const p1 = { x: toPxX(line.x1), y: toPxY(line.y1) };
    const p2 = { x: toPxX(line.x2), y: toPxY(line.y2) };
    const ang = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    ctx.save();
    ctx.translate(p1.x, p1.y);
    ctx.rotate(ang);
    const len = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    ctx.fillStyle = 'rgba(31,95,184,0.15)';
    ctx.fillRect(0, -t, len, t*2);
    ctx.restore();

    // linha principal (dashed leve)
    ctx.lineWidth = 3;
    ctx.setLineDash([10,8]);
    ctx.strokeStyle = '#1f5fb8';
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // alças só no hover/drag (ou quando forçado)
    if (hover || dragMode || forceHandles) {
      drawHandle(p1.x, p1.y, '#1f8b4c');
      drawHandle(p2.x, p2.y, '#b81f2b');
    }

    // sincroniza inputs
    x1I && (x1I.value = line.x1.toFixed(3));
    y1I && (y1I.value = line.y1.toFixed(3));
    x2I && (x2I.value = line.x2.toFixed(3));
    y2I && (y2I.value = line.y2.toFixed(3));
  }
  function drawHandle(x,y,color){
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI*2);
    ctx.fill();
  }

  // ===== Hit-tests =====
  const dist = (a,b)=> Math.hypot(a.x-b.x, a.y-b.y);
  const point = (x,y)=> ({x,y});
  function hitTest(mx,my){
    const p1 = point(toPxX(line.x1), toPxY(line.y1));
    const p2 = point(toPxX(line.x2), toPxY(line.y2));
    if (dist(point(mx,my), p1) <= 12) return 'p1';
    if (dist(point(mx,my), p2) <= 12) return 'p2';
    // distância ponto-linha
    const d = pointLineDistance(mx,my,p1,p2);
    if (d <= 10) return 'line';
    return null;
  }
  function pointLineDistance(px,py,a,b){
    const A = px - a.x, B = py - a.y, C = b.x - a.x, D = b.y - a.y;
    const dot = A*C + B*D;
    const len_sq = C*C + D*D || 1e-6;
    let t = dot / len_sq;
    t = Math.max(0, Math.min(1, t));
    const x = a.x + t*C, y = a.y + t*D;
    return Math.hypot(px-x, py-y);
  }

  // ===== Interação =====
  let dblTmp = null;
  canvas.addEventListener('mousemove', (e)=>{
    hover = true;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    document.body.style.cursor = hitTest(mx,my) ? 'pointer' : 'default';
  });
  canvas.addEventListener('mouseleave', ()=>{ hover = false; drawLine(); });

  // duplo clique define dois pontos
  canvas.addEventListener('dblclick', (e)=>{
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    if (!dblTmp) {
      dblTmp = {x,y};
    } else {
      const p1 = dblTmp, p2 = {x,y};
      line.x1 = toNrmX(p1.x); line.y1 = toNrmY(p1.y);
      line.x2 = toNrmX(p2.x); line.y2 = toNrmY(p2.y);
      dblTmp = null;
      saveLine();
      drawLine(true);
      scheduleAutoRestartIfMoved();
    }
  });

  canvas.addEventListener('mousedown', (e)=>{
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    dragMode = hitTest(mx,my);
    if (dragMode === 'line') {
      dragOffset.dx = mx - toPxX(line.x1);
      dragOffset.dy = my - toPxY(line.y1);
    }
  });
  window.addEventListener('mouseup', ()=> dragMode=null);
  window.addEventListener('mousemove', (e)=>{
    if (!dragMode) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;

    const prev = {...line};

    if (dragMode === 'p1') {
      line.x1 = clamp01(toNrmX(mx)); line.y1 = clamp01(toNrmY(my));
    } else if (dragMode === 'p2') {
      line.x2 = clamp01(toNrmX(mx)); line.y2 = clamp01(toNrmY(my));
    } else if (dragMode === 'line') {
      const nx1 = toNrmX(mx - dragOffset.dx), ny1 = toNrmY(my - dragOffset.dy);
      const dx = nx1 - line.x1, dy = ny1 - line.y1;
      line.x1 = clamp01(line.x1 + dx); line.y1 = clamp01(line.y1 + dy);
      line.x2 = clamp01(line.x2 + dx); line.y2 = clamp01(line.y2 + dy);
    }
    drawLine(true);
    if (movedEnough(prev, line)) {
      saveLine();
      scheduleAutoRestart();
    }
  });

  // Inputs manuais
  [x1I,y1I,x2I,y2I].forEach(inp=>{
    inp?.addEventListener('input', ()=>{
      const prev = {...line};
      line.x1 = clamp01(parseFloat(x1I.value));
      line.y1 = clamp01(parseFloat(y1I.value));
      line.x2 = clamp01(parseFloat(x2I.value));
      line.y2 = clamp01(parseFloat(y2I.value));
      drawLine(true);
      saveLine();
      if (movedEnough(prev, line)) scheduleAutoRestart();
    });
  });
  [sampleFpsI, chunkI, workersI, toleranceI].forEach(inp=>{
    inp?.addEventListener('change', ()=>{
      saveParams();
      if (!currentVideoPath || !autoStreaming) return;
      stopStream();
      processStream();
    });
  });

  // Atalhos: setas movem p2; Alt = p1; Shift = passo 2x; H = travar horizontal
  let lockHorizontal = false;
  window.addEventListener('keydown', (e)=>{
    const step = (e.shiftKey? 0.02 : 0.01);
    let changed = false;
    if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) {
      const tgt = e.altKey ? 'p1' : 'p2';
      if (tgt === 'p2') {
        if (e.key==='ArrowLeft')  { line.x2 = clamp01(line.x2 - step); changed=true; }
        if (e.key==='ArrowRight') { line.x2 = clamp01(line.x2 + step); changed=true; }
        if (!lockHorizontal){
          if (e.key==='ArrowUp')   { line.y2 = clamp01(line.y2 - step); changed=true; }
          if (e.key==='ArrowDown') { line.y2 = clamp01(line.y2 + step); changed=true; }
        }
      } else {
        if (e.key==='ArrowLeft')  { line.x1 = clamp01(line.x1 - step); changed=true; }
        if (e.key==='ArrowRight') { line.x1 = clamp01(line.x1 + step); changed=true; }
        if (!lockHorizontal){
          if (e.key==='ArrowUp')   { line.y1 = clamp01(line.y1 - step); changed=true; }
          if (e.key==='ArrowDown') { line.y1 = clamp01(line.y1 + step); changed=true; }
        }
      }
      if (changed) { drawLine(true); saveLine(); scheduleAutoRestart(); }
    }
    if (e.key.toLowerCase() === 'h') {
      lockHorizontal = !lockHorizontal;
      if (lockHorizontal) {
        // nivela horizontal usando y médio
        const ymid = (line.y1 + line.y2)/2;
        line.y1 = line.y2 = ymid;
        drawLine(true); saveLine(); scheduleAutoRestart();
        toast('Linha travada horizontalmente.', 'info');
      } else {
        toast('Travamento horizontal desabilitado.', 'info');
      }
    }
    if (e.key === 'Escape') {
      stopStream();
      autoStreaming = false;
      lockUI(false);
      setStatus('Parado');
    }
  });

  // ===== SSE helpers =====
  function stopStream(){
    if (currentEventSource){
      try { currentEventSource.close(); } catch(e){}
      currentEventSource = null;
    }
  }
  function startStreamWithParams(params){
    stopStream();
    const es = new EventSource(`/process/stream?${params.toString()}`);
    currentEventSource = es;
    return es;
  }

  // ===== Lógica de processamento =====
  async function processBatch(){
    if (!currentVideoPath) return toast('Faça o upload de um vídeo primeiro.', 'warn');
    lockUI(true); setStatus('Processando…', 0);
    try{
      const body = {
        video_path: currentVideoPath,
        line: [line.x1, line.y1, line.x2, line.y2],
        sample_fps: parseFloat(sampleFpsI?.value || '5'),
        chunk_seconds: parseInt(chunkI?.value || '60'),
        workers: parseInt(workersI?.value || '0'),
        save_annotated: !!saveAnnI?.checked
      };
      const res = await safeFetch('/process', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(res.error || 'Erro no processamento.');
      populateResults(res);
      setStatus('Concluído', 100);
      toast('Processamento concluído.', 'ok');
    } catch(err){
      setStatus('Erro'); toast(err.message, 'err');
    } finally {
      lockUI(false);
    }
  }

  function processStream(){
    if (!currentVideoPath) return toast('Faça o upload de um vídeo primeiro.', 'warn');
    lockUI(true); setStatus('Iniciando streaming…', 0);

    // reset taxa/min ao iniciar um novo stream
    firstProgressTs = null; lastProgressTs = null;
    sseRetry.tries = 0;

    const params = new URLSearchParams({
      video_path: currentVideoPath,
      x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2,
      sample_fps: sampleFpsI?.value || '5',
      chunk_seconds: chunkI?.value || '60',
      workers: workersI?.value || '0'
    });

    const es = startStreamWithParams(params);

    es.onmessage = (e)=>{
      try{
        const data = JSON.parse(e.data);
        if (data.type === 'progress'){
          // taxa/min calculada a partir dos deltas e tempo
          const now = Date.now();
          if (!firstProgressTs) firstProgressTs = now;
          if (!lastProgressTs) lastProgressTs = now;
          const dtSec = Math.max(0.001, (now - lastProgressTs)/1000);

          // deltas visuais
          const din = Math.max(0, (data.in_partial ?? 0) - lastPartial.in);
          const dout = Math.max(0, (data.out_partial ?? 0) - lastPartial.out);
          if (din>0 || dout>0) flashCounts(din, dout);

          lastPartial.in = data.in_partial ?? lastPartial.in;
          lastPartial.out = data.out_partial ?? lastPartial.out;

          // taxa por minuto (média simples no intervalo)
          const rpmIn = (din/dtSec)*60;
          const rpmOut = (dout/dtSec)*60;

          setStatus(
            `Processando… ${data.pct}% | IN ${lastPartial.in} • OUT ${lastPartial.out} | +/min ${rpmIn.toFixed(1)}/${rpmOut.toFixed(1)}`,
            data.pct
          );
          inCount.textContent = lastPartial.in;
          outCount.textContent = lastPartial.out;
          netCount.textContent = (lastPartial.in - lastPartial.out);

          lastProgressTs = now;
        } else if (data.type === 'done') {
          populateResults(data);
          setStatus('Concluído', 100);
          es.close(); currentEventSource = null;
          lockUI(false);
          toast(autoStreaming ? 'Contagem concluída.' : 'Streaming finalizado.', 'ok');
          autoStreaming = false;
        } else if (data.type === 'error') {
          setStatus('Erro'); toast(data.message || 'Erro no streaming.', 'err');
          es.close(); currentEventSource = null; lockUI(false);
          maybeReconnect();
        }
      }catch(err){
        setStatus('Erro'); toast(err.message, 'err');
        es.close(); currentEventSource = null; lockUI(false);
        maybeReconnect();
      }
    };
    es.onerror = ()=>{
      setStatus('Erro na conexão SSE'); toast('Erro na conexão SSE.', 'err');
      es.close(); currentEventSource = null; lockUI(false);
      maybeReconnect();
    };
  }

  function maybeReconnect(){
    if (!autoStreaming) return;
    if (sseRetry.tries >= sseRetry.max) {
      autoStreaming = false;
      toast('Não foi possível reconectar ao streaming.', 'err');
      return;
    }
    const delay = Math.min(5000, 400 * Math.pow(2, sseRetry.tries++));
    setTimeout(()=>{ lockUI(false); processStream(); }, delay);
  }

  function populateResults(payload){
    const { in_total, out_total, net_total, windows, csv_path, annotated_path } = payload;
    // se vier total, sobrescreve os parciais
    if (Number.isFinite(in_total)) inCount.textContent = in_total;
    if (Number.isFinite(out_total)) outCount.textContent = out_total;
    if (Number.isFinite(net_total)) netCount.textContent = net_total;

    downloads.innerHTML = '';
    if (csv_path){
      const a = document.createElement('a');
      a.href = `/download/${encodeURIComponent(csv_path.split('outputs/').pop())}`;
      a.className = 'btn small';
      a.textContent = 'Baixar CSV';
      downloads.appendChild(a);
    }
    if (annotated_path){
      const a2 = document.createElement('a');
      a2.href = `/download/${encodeURIComponent(annotated_path.split('outputs/').pop())}`;
      a2.className = 'btn small ghost';
      a2.textContent = 'Baixar MP4 anotado';
      downloads.appendChild(a2);
    }

    windowsTable.innerHTML = '';
    (windows || []).forEach(win=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${win.start}</td><td>${win.end}</td><td>${win.in}</td><td>${win.out}</td>`;
      windowsTable.appendChild(tr);
    });
  }

  async function safeFetch(url, opts={}){
    const r = await fetch(url, opts);
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('application/json')){
      const j = await r.json();
      if (!r.ok || j.ok === false) return { ok:false, ...j };
      return j;
    } else {
      if (!r.ok) return { ok:false, error: r.statusText || 'Erro HTTP' };
      return { ok:true };
    }
  }

  // ===== Botões =====
  btnProcess?.addEventListener('click', ()=>{
    stopStream();
    autoStreaming = false;
    processBatch();
  });
  btnStream?.addEventListener('click', ()=>{
    autoStreaming = false;
    stopStream();
    processStream();
  });
  btnReset?.addEventListener('click', ()=>{
    lastPartial = { in:0, out:0 };
    inCount.textContent = 0; outCount.textContent = 0; netCount.textContent = 0;
    setStatus('Contadores zerados');
  });

  // ===== Calculadora (Micro e Mid) =====
  function calcular(precoLitro, km, lpkm, manutKm,
                    outCons, outCComb, outCKm, outTotal){
    km = parseFloat(km)||0; lpkm = parseFloat(lpkm)||0; precoLitro = parseFloat(precoLitro)||0; manutKm = parseFloat(manutKm)||0;
    const consumo = km * lpkm;
    const custoComb = consumo * precoLitro;
    const custoTotal = (km * manutKm) + custoComb;
    const ckm = km > 0 ? (custoTotal / km) : 0;
    el(outCons).textContent = (consumo.toFixed(2));
    el(outCComb).textContent = formatBRL(custoComb);
    el(outCKm).textContent = formatBRL(ckm);
    el(outTotal).textContent = formatBRL(custoTotal);
  }
  window.calcularMicro = () => calcular(
    el('precoLitro').value,
    el('kmRodadosMicro').value,
    el('litrosPorKmMicro').value,
    el('custoPorKmManutencaoMicro').value,
    'consumoTotalMicro', 'custoTotalCombustivelMicro', 'custoPorKmMicro', 'custoTotalMicro'
  );
  window.calcularMid = () => calcular(
    el('precoLitroMid').value,
    el('kmRodadosMid').value,
    el('litrosPorKmMid').value,
    el('custoPorKmManutencaoMid').value,
    'consumoTotalMid', 'custoTotalCombustivelMid', 'custoPorKmMid', 'custoTotalMid'
  );

  // ===== Reinício automático inteligente =====
  function scheduleAutoRestart(){
    if (!autoStreaming) return;
    if (rearmTimer) clearTimeout(rearmTimer);
    rearmTimer = setTimeout(()=>{
      stopStream();
      setStatus('Recontando…', 0);
      processStream();
    }, 450);
  }

  // só reinicia se movimento exceder 0.5% da tela (evita jitter)
  function movedEnough(prev, curr){
    const dx = Math.abs(prev.x1 - curr.x1) + Math.abs(prev.x2 - curr.x2);
    const dy = Math.abs(prev.y1 - curr.y1) + Math.abs(prev.y2 - curr.y2);
    return (dx + dy) >= 0.005; // ~0,5% total
  }
  function scheduleAutoRestartIfMoved(){ scheduleAutoRestart(); }

  // ===== Efeitos visuais de delta =====
  function flashCounts(din, dout, reset=false){
    if (reset){ inCount.classList.remove('pulse'); outCount.classList.remove('pulse'); return; }
    if (din>0){ pulse(inCount); }
    if (dout>0){ pulse(outCount); }
    function pulse(node){
      node.classList.remove('pulse');
      // força reflow pra reiniciar animação
      // eslint-disable-next-line no-unused-expressions
      node.offsetWidth;
      node.classList.add('pulse');
      setTimeout(()=> node.classList.remove('pulse'), 500);
    }
  }

  // ===== Throttle util =====
  function throttle(fn, ms){
    let t=0, lastArgs=null, queued=false;
    return (...args)=>{
      const now = Date.now();
      if (now - t >= ms){
        t = now; fn(...args);
      } else {
        lastArgs = args;
        if (!queued){
          queued = true;
          setTimeout(()=>{
            queued = false;
            t = Date.now();
            fn(...(lastArgs||[]));
          }, ms - (now - t));
        }
      }
    };
  }

  // ===== Inicialização =====
  syncCanvasSize();
  drawLine(true);

''})();
  