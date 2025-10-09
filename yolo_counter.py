# yolo_counter.py
# Contagem por cruzamento de linha (normalizada) com YOLO (se disponível) + fallback.
# Contrato esperado pelo app.py:
# - process_stream(video_path, line|line_norm, sample_fps, chunk_seconds, workers) -> yield dicts {type:'progress'|'done'|'error', ...}
# - process_video(video_path, line|line_norm, sample_fps, chunk_seconds, workers, save_annotated) -> dict com totais, windows, paths

from __future__ import annotations
import os, csv, time, math, pathlib, datetime as dt
from typing import Dict, Tuple, Generator, List, Optional

try:
    import cv2
    import numpy as np
except Exception as e:
    raise RuntimeError("OpenCV (cv2) é obrigatório. Instale com: pip install opencv-python") from e

# YOLO opcional
YOLO_AVAILABLE = False
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
    YOLO = _YOLO
except Exception:
    YOLO_AVAILABLE = False
    YOLO = None

# ---------- helpers de coerção ----------
def _coerce_line(val) -> Optional[Tuple[float,float,float,float]]:
    """
    Aceita tuple/list/str e retorna (x1,y1,x2,y2) float em [0..1] ou None.
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            t = tuple(float(x) for x in val)
            if all(0.0 <= v <= 1.0 for v in t):
                return t
            # se vier em pixels por engano, normaliza depois — aqui só devolve mesmo
            return t
        except Exception:
            return None
    if isinstance(val, str):
        try:
            parts = [p.strip() for p in val.split(",")]
            if len(parts) == 4:
                t = tuple(float(x) for x in parts)
                return t
        except Exception:
            return None
    return None

def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _coerce_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)

# ---------- Utilidades geométricas ----------
def line_side(x, y, x1, y1, x2, y2) -> float:
    """Sinal do produto vetorial: >0 um lado, <0 outro lado, ~0 em cima."""
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def normalize_line_to_pixels(line_norm, w, h) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = line_norm
    # se vierem valores >1, assume que já estão em pixels
    if max(abs(x1),abs(y1),abs(x2),abs(y2)) > 1.0001:
        return int(x1), int(y1), int(x2), int(y2)
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

def ensure_dirs(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---------- Núcleo de contagem ----------
class LineCounter:
    """
    Conta cruzamentos por track_id: quando o sinal do lado muda, incrementa IN/OUT.
    Direção: prev<0->side>0 => IN ; prev>0->side<0 => OUT (heurística estável).
    """
    def __init__(self, w:int, h:int, line_norm:Tuple[float,float,float,float]):
        self.w, self.h = w, h
        self.x1, self.y1, self.x2, self.y2 = normalize_line_to_pixels(line_norm, w, h)
        self.last_side: Dict[int, float] = {}  # track_id -> side
        self.in_count = 0
        self.out_count = 0

    def update_point(self, track_id:int, cx:float, cy:float):
        side = line_side(cx, cy, self.x1, self.y1, self.x2, self.y2)
        prev = self.last_side.get(track_id)
        if prev is not None:
            if prev == 0: prev = -1e-6
            if side == 0: side = 1e-6
            if prev * side < 0:
                if prev < 0 < side:
                    self.in_count += 1
                elif prev > 0 > side:
                    self.out_count += 1
        self.last_side[track_id] = side

# ---------- Pipeline principal ----------
def _iterate_frames(cap, every_n_frames:int):
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % every_n_frames == 0:
            yield idx, frame
        ok, frame = cap.read()
        idx += 1

def _estimate_every_n_frames(fps: float, sample_fps: float) -> int:
    if fps <= 0: fps = 25.0
    if sample_fps <= 0: sample_fps = 5.0
    step = max(1, int(round(fps / sample_fps)))
    return step

def _draw_overlays(frame, counter: LineCounter, in_partial:int, out_partial:int):
    # linha
    cv2.line(frame, (counter.x1, counter.y1), (counter.x2, counter.y2), (184,95,31), 2)  # BGR
    # contadores
    txt = f"IN: {in_partial}  OUT: {out_partial}  NET: {in_partial - out_partial}"
    cv2.rectangle(frame, (10,10), (10+320, 45), (0,0,0), -1)
    cv2.putText(frame, txt, (18,38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return frame

def _run_yolo_on_frame(model, frame):
    # Pessoas (classe 0)
    res = model.predict(source=frame, classes=[0], conf=0.25, verbose=False)
    if not res: return []
    r = res[0]
    dets = []
    if getattr(r, "boxes", None) is None: return dets
    boxes = r.boxes.xyxy.cpu().numpy()
    for i, (x1,y1,x2,y2) in enumerate(boxes):
        cx = float((x1+x2)/2.0)
        cy = float((y1+y2)/2.0)
        dets.append({"id": i, "cx": cx, "cy": cy})
    return dets

def _run_yolo_track_frames(model, frames_iter, counter: LineCounter, writer=None):
    for idx, frame in frames_iter:
        dets = _run_yolo_on_frame(model, frame)
        for d in dets:
            counter.update_point(d["id"], d["cx"], d["cy"])
        if writer is not None:
            _draw_overlays(frame, counter, counter.in_count, counter.out_count)
            writer.write(frame)
        yield idx

def _fallback_dummy(frames_iter, counter: LineCounter, writer=None):
    for idx, frame in frames_iter:
        if writer is not None:
            _draw_overlays(frame, counter, counter.in_count, counter.out_count)
            writer.write(frame)
        yield idx

# ---------- API esperada pelo app ----------
def process_stream(
    video_path: str,
    line: Tuple[float,float,float,float] | None = None,
    sample_fps: float = 5.0,
    chunk_seconds: int = 60,
    workers: int = 0,
    line_norm: Tuple[float,float,float,float] | None = None,
    **kwargs
) -> Generator[dict, None, None]:
    """
    Eventos:
      {"type":"progress","pct":int,"in_partial":int,"out_partial":int}
      {"type":"done","in_total":int,"out_total":int,"net_total":int,"windows":[...]}
      {"type":"error","message":"..."}
    """
    try:
        if not os.path.exists(video_path):
            yield {"type":"error","message":"Vídeo não encontrado."}
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {"type":"error","message":"Falha ao abrir vídeo."}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # coersões de tipos (querystring chega como str)
        sample_fps = _coerce_float(sample_fps, 5.0)
        chunk_seconds = _coerce_int(chunk_seconds, 60)
        workers = _coerce_int(workers, 0)

        # aceita line OU line_norm; ambos podem vir como string "x1,y1,x2,y2"
        line = _coerce_line(line if line is not None else line_norm)
        if line is None:
            yield {"type":"error","message":"Linha inválida (esperado x1,y1,x2,y2 normalizados)."}
            return

        step = _estimate_every_n_frames(fps, sample_fps)
        counter = LineCounter(w, h, line)
        frames_iter = _iterate_frames(cap, step)

        if YOLO_AVAILABLE:
            model = YOLO("yolov8n.pt")  # leve para CPU
            runner = _run_yolo_track_frames(model, frames_iter, counter, writer=None)
        else:
            runner = _fallback_dummy(frames_iter, counter, writer=None)

        last_emit = 0.0
        for idx in runner:
            pct = int(min(100, math.floor((idx+1)/max(1,total_frames)*100)))
            now = time.time()
            if now - last_emit > 0.08:  # ~10 Hz
                yield {
                    "type":"progress",
                    "pct": pct,
                    "in_partial": int(counter.in_count),
                    "out_partial": int(counter.out_count)
                }
                last_emit = now

        net_total = int(counter.in_count - counter.out_count)
        dur_s = int(total_frames / (fps or 1))
        windows = [{
            "start": "00:00:00",
            "end": _fmt_s(dur_s),
            "in": int(counter.in_count),
            "out": int(counter.out_count),
        }]
        yield {
            "type":"done",
            "in_total": int(counter.in_count),
            "out_total": int(counter.out_count),
            "net_total": net_total,
            "windows": windows
        }

    except Exception as e:
        yield {"type":"error","message": f"{type(e).__name__}: {e}"}

def process_video(
    video_path: str,
    line: Tuple[float,float,float,float] | None = None,
    sample_fps: float = 5.0,
    chunk_seconds: int = 60,
    workers: int = 0,
    save_annotated: bool = False,
    line_norm: Tuple[float,float,float,float] | None = None,
    **kwargs
) -> dict:
    """
    Retorna:
      { ok, in_total, out_total, net_total, windows, csv_path, annotated_path }
    """
    if not os.path.exists(video_path):
        return {"ok": False, "error": "Vídeo não encontrado."}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Falha ao abrir vídeo."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # coersões
    sample_fps = _coerce_float(sample_fps, 5.0)
    chunk_seconds = _coerce_int(chunk_seconds, 60)
    workers = _coerce_int(workers, 0)
    save_annotated = bool(save_annotated)

    line = _coerce_line(line if line is not None else line_norm)
    if line is None:
        return {"ok": False, "error": "Linha inválida (esperado x1,y1,x2,y2 normalizados)."}

    step = _estimate_every_n_frames(fps, sample_fps)

    # Saídas
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = os.path.join("outputs", stamp)
    csv_path = os.path.join(base_out, "contagem.csv")
    annotated_path = None

    writer = None
    if save_annotated:
        ensure_dirs(os.path.join(base_out, "video.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(base_out, "video.mp4"),
                                 fourcc,
                                 max(5.0, min(30.0, fps/step)),
                                 (w, h))
        annotated_path = os.path.join(base_out, "video.mp4")

    counter = LineCounter(w, h, line)
    frames_iter = _iterate_frames(cap, step)

    if YOLO_AVAILABLE:
        model = YOLO("yolov8n.pt")
        runner = _run_yolo_track_frames(model, frames_iter, counter, writer=writer)
    else:
        runner = _fallback_dummy(frames_iter, counter, writer=writer)

    # janelas simples por chunk_seconds
    win_list: List[dict] = []
    cur_start = 0.0
    last_in = 0
    last_out = 0

    for idx in runner:
        t = (idx / (fps or 1.0))
        if (t - cur_start) >= max(1, int(chunk_seconds)):
            win_list.append({
                "start": _fmt_s(int(cur_start)),
                "end": _fmt_s(int(t)),
                "in": int(counter.in_count - last_in),
                "out": int(counter.out_count - last_out),
            })
            cur_start = t
            last_in = counter.in_count
            last_out = counter.out_count

    total_secs = int(total_frames / (fps or 1.0))
    if (counter.in_count - last_in) != 0 or (counter.out_count - last_out) != 0 or not win_list:
        win_list.append({
            "start": _fmt_s(int(cur_start)),
            "end": _fmt_s(int(total_secs)),
            "in": int(counter.in_count - last_in),
            "out": int(counter.out_count - last_out),
        })

    if writer is not None:
        writer.release()

    ensure_dirs(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f, delimiter=';')
        wcsv.writerow(["start","end","in","out"])
        for wrow in win_list:
            wcsv.writerow([wrow["start"], wrow["end"], wrow["in"], wrow["out"]])

    return {
        "ok": True,
        "in_total": int(counter.in_count),
        "out_total": int(counter.out_count),
        "net_total": int(counter.in_count - counter.out_count),
        "windows": win_list,
        "csv_path": csv_path,
        "annotated_path": annotated_path
    }

# ---------- helpers ----------
def _fmt_s(s: int) -> str:
    s = int(max(0, s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"
