const $ = (sel) => document.querySelector(sel);
const statusBox = $("#status");
const resultsBox = $("#results");
const inCount = $("#inCount");
const outCount = $("#outCount");
const windowsWrap = $("#windowsWrap");
const windowsTableBody = $("#windowsTable tbody");
const downloads = $("#downloads");
const annotatedWrap = $("#annotatedWrap");
const annotatedVideo = $("#annotatedVideo");

const videoInput = $("#video");
const videoEl = $("#videoPreview");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const hint = document.querySelector(".overlay-hint");

const x1 = $("#x1"), y1 = $("#y1"), x2 = $("#x2"), y2 = $("#y2");
const btnProcess = $("#btnProcess");
const btnStream = $("#btnStream");

let clickState = 0; // 0->aguardando p1, 1->aguardando p2
let naturalWidth = 0, naturalHeight = 0;

function setStatus(msg){ statusBox.textContent = msg; }
function toggleBusy(isBusy){
  btnProcess.disabled = isBusy;
  btnStream.disabled = isBusy;
}

function drawLinePreview(){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if (!naturalWidth || !naturalHeight) return;

  const p1 = {
    x: parseFloat(x1.value) * overlay.width,
    y: parseFloat(y1.value) * overlay.height
  };
  const p2 = {
    x: parseFloat(x2.value) * overlay.width,
    y: parseFloat(y2.value) * overlay.height
  };

  // linha "grossa"
  ctx.strokeStyle = "rgba(255,215,0,0.95)";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.stroke();

  // alças
  ctx.fillStyle = "#0a2f6b";
  [p1,p2].forEach(p=>{
    ctx.beginPath(); ctx.arc(p.x,p.y,6,0,Math.PI*2); ctx.fill();
  });
}

function resizeOverlay(){
  const rect = videoEl.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
  drawLinePreview();
}

function setPointsFromClick(ev){
  if (!naturalWidth || !naturalHeight) return;
  const r = overlay.getBoundingClientRect();
  const nx = (ev.clientX - r.left) / r.width;
  const ny = (ev.clientY - r.top) / r.height;

  if (clickState === 0){
    x1.value = clamp(nx,0,1).toFixed(3);
    y1.value = clamp(ny,0,1).toFixed(3);
    clickState = 1;
    hint.textContent = "Agora clique no segundo ponto da linha";
  } else {
    x2.value = clamp(nx,0,1).toFixed(3);
    y2.value = clamp(ny,0,1).toFixed(3);
    clickState = 0;
    hint.textContent = "Clique 2x para redefinir a linha";
  }
  drawLinePreview();
}

function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }

["input","change"].forEach(evt=>{
  [x1,y1,x2,y2].forEach(el=>el.addEventListener(evt, drawLinePreview));
});
window.addEventListener("resize", resizeOverlay);
overlay.addEventListener("click", setPointsFromClick);

// Preview do vídeo selecionado
videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  videoEl.src = url;
  videoEl.onloadedmetadata = () => {
    naturalWidth = videoEl.videoWidth;
    naturalHeight = videoEl.videoHeight;
    resizeOverlay();
  };
});

// ---- Submissões ----
async function submitCommon(stream = false){
  const file = videoInput.files?.[0];
  if(!file){ setStatus("Selecione um vídeo para processar."); return; }

  const form = new FormData();
  form.append("video", file);
  form.append("x1", x1.value);
  form.append("y1", y1.value);
  form.append("x2", x2.value);
  form.append("y2", y2.value);
  form.append("sample_fps", $("#sample_fps").value);
  form.append("save_annotated", $("#save_annotated").checked ? "1" : "0");
  form.append("chunk_seconds", $("#chunk_seconds").value);
  form.append("workers", $("#workers").value);

  resultsBox.classList.add("hidden");
  annotatedWrap.classList.add("hidden");
  windowsWrap.classList.add("hidden");
  downloads.innerHTML = "";
  inCount.textContent = "0";
  outCount.textContent = "0";

  toggleBusy(true);
  setStatus(stream ? "Processando em streaming…" : "Processando…");

  try{
    if(!stream){
      const resp = await fetch("/upload",{ method:"POST", body: form });
      const data = await resp.json();
      handleFinal(data);
    }else{
      // streaming
      // parâmetros extras
      form.append("window_minutes","5");
      form.append("clock_start","");

      const resp = await fetch("/upload_stream",{ method:"POST", body: form });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while(true){
        const {value, done} = await reader.read();
        if(done) break;
        buf += decoder.decode(value, {stream:true});
        let lines = buf.split("\n");
        buf = lines.pop() || "";
        for(const line of lines){
          if(!line.trim()) continue;
          const ev = JSON.parse(line);
          if(ev.event === "start"){
            setStatus("Iniciado…");
          }else if(ev.event === "chunk_result"){
            setStatus(`Chunk ${ev.index}: IN ${ev.in} / OUT ${ev.out}`);
          }else if(ev.event === "final"){
            handleFinal(ev);
          }else if(ev.event === "error"){
            setStatus("Erro: " + ev.detail);
          }
        }
      }
    }
  }catch(e){
    setStatus("Erro ao enviar/processar: " + e.message);
  }finally{
    toggleBusy(false);
  }
}

function handleFinal(payload){
  if(payload.error){
    setStatus("Erro: " + (payload.detail || payload.error));
    return;
  }
  setStatus("Concluído!");
  resultsBox.classList.remove("hidden");

  if(payload.totals){
    inCount.textContent = payload.totals.in ?? 0;
    outCount.textContent = payload.totals.out ?? 0;
  }

  downloads.innerHTML = "";
  if(payload.annotated_video_url){
    const a = document.createElement("a");
    a.href = payload.annotated_video_url;
    a.textContent = "Baixar vídeo anotado";
    downloads.appendChild(a);

    annotatedVideo.src = payload.annotated_video_url;
    annotatedWrap.classList.remove("hidden");
  }
  if(Array.isArray(payload.annotated_segment_urls)){
    payload.annotated_segment_urls.forEach((u,idx)=>{
      const a = document.createElement("a");
      a.href = u;
      a.textContent = `Segmento ${idx+1}`;
      a.classList.add("secondary");
      downloads.appendChild(a);
    });
  }

  if(Array.isArray(payload.windows) && payload.windows.length){
    const tbody = windowsTableBody;
    tbody.innerHTML = "";
    payload.windows.forEach(w=>{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${w.label}</td><td>${w.in}</td><td>${w.out}</td>`;
      tbody.appendChild(tr);
    });
    windowsWrap.classList.remove("hidden");
  }
}

btnProcess.addEventListener("click", ()=> submitCommon(false));
btnStream.addEventListener("click", ()=> submitCommon(true));


function flashHint(ms=8000){
  const hint = document.querySelector('.overlay-hint');
  if(!hint) return;
  hint.style.opacity = '1';
  clearTimeout(flashHint._t);
  flashHint._t = setTimeout(()=>{ hint.style.opacity = '0'; }, ms);
}

// dentro do onloadedmetadata, depois de drawLinePreview():
flashHint();

// quando o usuário mexer na linha, esconda a dica:
overlay.addEventListener('mousedown', ()=> flashHint(0));
overlay.addEventListener('touchstart', ()=> flashHint(0), {passive:true});

function calcularMicro() {
  const kmRodadosMicro = parseFloat(document.getElementById('kmRodadosMicro').value);
  const litrosPorKmMicro = parseFloat(document.getElementById('litrosPorKmMicro').value);
  const precoLitro = parseFloat(document.getElementById('precoLitro').value);
  const custoPorKmManutencaoMicro = parseFloat(document.getElementById('custoPorKmManutencaoMicro').value);

  if (isNaN(kmRodadosMicro) || isNaN(litrosPorKmMicro) || isNaN(precoLitro) || isNaN(custoPorKmManutencaoMicro)) {
    alert('Por favor, preencha todos os campos corretamente.');
    return;
  }

  const consumoTotalMicro = kmRodadosMicro * litrosPorKmMicro;
  const custoTotalCombustivelMicro = consumoTotalMicro * precoLitro;
  const custoPorKmMicro = custoTotalCombustivelMicro / kmRodadosMicro;
  const custoTotalMicro = custoTotalCombustivelMicro + (custoPorKmManutencaoMicro * kmRodadosMicro);

  document.getElementById('consumoTotalMicro').textContent = consumoTotalMicro.toFixed(2) + ' L';
  document.getElementById('custoTotalCombustivelMicro').textContent = 'R$ ' + custoTotalCombustivelMicro.toFixed(2);
  document.getElementById('custoPorKmMicro').textContent = 'R$ ' + custoPorKmMicro.toFixed(2);
  document.getElementById('custoTotalMicro').textContent = 'R$ ' + custoTotalMicro.toFixed(2);
}


function calcularMid() {
  const kmRodadosMid = parseFloat(document.getElementById('kmRodadosMid').value);
  const litrosPorKmMid = parseFloat(document.getElementById('litrosPorKmMid').value);
  const precoLitro = parseFloat(document.getElementById('precoLitro').value);
  const custoPorKmManutencaoMid = parseFloat(document.getElementById('custoPorKmManutencaoMid').value);

  if (isNaN(kmRodadosMid) || isNaN(litrosPorKmMid) || isNaN(precoLitro) || isNaN(custoPorKmManutencaoMid)) {
    alert('Por favor, preencha todos os campos corretamente.');
    return;
  }

  const consumoTotalMid = kmRodadosMid * litrosPorKmMid;
  const custoTotalCombustivelMid = consumoTotalMid * precoLitro;
  const custoPorKmMid = custoTotalCombustivelMid / kmRodadosMid;
  const custoTotalMid = custoTotalCombustivelMid + (custoPorKmManutencaoMid * kmRodadosMid);

  document.getElementById('consumoTotalMid').textContent = consumoTotalMid.toFixed(2) + ' L';
  document.getElementById('custoTotalCombustivelMid').textContent = 'R$ ' + custoTotalCombustivelMid.toFixed(2);
  document.getElementById('custoPorKmMid').textContent = 'R$ ' + custoPorKmMid.toFixed(2);
  document.getElementById('custoTotalMid').textContent = 'R$ ' + custoTotalMid.toFixed(2);
}