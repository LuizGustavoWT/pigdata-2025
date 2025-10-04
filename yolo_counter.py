# yolo_counter.py
import os
from pathlib import Path
import cv2
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from ultralytics import YOLO

from utils import (
    iou_xywh, point_side_of_line, signed_distance_to_line,
    segments_intersect, put_text, draw_thick_line, Track
)
from utils_ffmpeg import split_into_chunks, concat_videos_mp4, probe_duration
from cpu_tunning import tune_cpu_threads

# =================== Modelo ===================
@lru_cache(maxsize=1)
def get_model_ultra(weights="yolov8s.pt"):
    # Para CPU, o 's' costuma dar ganho de precisão vs 'n' (ainda leve).
    m = YOLO(weights)
    return m

# =================== Núcleo por segmento ===================
def process_segment(video_path, line_norm, sample_fps, save_annotated, out_dir, time_offset=0.0, suffix=""):
    tune_cpu_threads(num_infer_threads=4, num_interop_threads=1, opencv_threads=1)
    model = get_model_ultra("yolov8s.pt")  # troque para 'n' se ficar pesado

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # linha em px
    x1n,y1n,x2n,y2n = line_norm
    x1,y1 = int(x1n*width),  int(y1n*height)
    x2,y2 = int(x2n*width),  int(y2n*height)

    step = max(1, int(round(src_fps / float(sample_fps))))
    writer = None
    annotated_path = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_path = str(Path(out_dir) / (Path(video_path).stem + f"{suffix}_annotated.mp4"))
        writer = cv2.VideoWriter(annotated_path, fourcc, max(float(sample_fps), 5.0), (width, height))

    # ---- Rastreamento & parâmetros anti-ruído ----
    next_id = 1
    tracks = {}  # tid -> Track
    max_misses = 12
    iou_th = 0.3
    # Para contagem
    min_disp_px = max(4, int(0.004 * (width + height)))   # deslocamento mínimo para considerar cruzamento
    cooldown_frames = int(sample_fps * 0.75)              # evita dupla contagem se ficar “quicando” na linha

    total_in = total_out = 0
    timeline = []
    frame_idx = -1
    out_idx = -1

    # vetor normal à linha (para assinar direção via produto escalar)
    dx, dy = (x2 - x1, y2 - y1)
    nx, ny = (dy, -dx)  # perpendicular

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        out_idx += 1
        ts = out_idx / float(sample_fps) + float(time_offset)

        # ---------- Detecção ----------
        # classes=[0] -> person
        results = model(frame, imgsz=640, classes=[0], conf=0.35, iou=0.6, verbose=False)
        dets = []
        if len(results):
            b = results[0].boxes
            if b is not None and b.xywh is not None:
                xywh = b.xywh.cpu().numpy()
                scores = b.conf.cpu().numpy()
                for (cx, cy, w, h), sc in zip(xywh, scores):
                    if sc < 0.35:
                        continue
                    x = float(cx - w/2); y = float(cy - h/2)
                    dets.append([x, y, float(w), float(h)])

        # ---------- Associação por IoU ----------
        # marca todos como não atualizados
        for trk in tracks.values():
            trk.misses += 1  # envelhece por padrão (se atualizar, voltamos a 0)

        used = set()
        for db in dets:
            # encontra melhor track
            best_tid, best_iou = None, 0.0
            for tid, trk in tracks.items():
                if tid in used:  # um det por track
                    continue
                score = iou_xywh(db, trk.bbox)
                if score > best_iou:
                    best_iou, best_tid = score, tid
            if best_tid is not None and best_iou >= iou_th:
                trk = tracks[best_tid]
                trk.update(db)
                trk.misses = 0
                used.add(best_tid)
            else:
                # cria novo track
                cx = db[0] + db[2]/2.0
                cy = db[1] + db[3]/2.0
                side = 1 if point_side_of_line(cx, cy, x1, y1, x2, y2) > 0 else -1
                tracks[next_id] = Track(next_id, db, side)
                used.add(next_id)
                next_id += 1

        # remove tracks perdidos
        to_del = [tid for tid,trk in tracks.items() if trk.misses > max_misses]
        for tid in to_del:
            del tracks[tid]

        # ---------- Contagem por cruzamento de segmento ----------
        crossed_in = 0
        crossed_out = 0

        for tid, trk in list(tracks.items()):
            if len(trk.history) < 2:
                continue

            p_prev = trk.history[-2]
            p_now  = trk.history[-1]

            # deslocamento mínimo para evitar ruído
            if abs(p_now[0] - p_prev[0]) + abs(p_now[1] - p_prev[1]) < min_disp_px:
                continue

            # checa se o segmento do track cruza a linha
            if not segments_intersect((x1,y1),(x2,y2), p_prev, p_now):
                # também aceita troca de sinal da distância assinada (mais robusto)
                d1 = signed_distance_to_line(p_prev[0], p_prev[1], x1, y1, x2, y2)
                d2 = signed_distance_to_line(p_now[0],  p_now[1],  x1, y1, x2, y2)
                if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
                    continue

            # cooldown para não contar 2x
            if trk.alive_frames - trk.last_cross_frame < cooldown_frames:
                continue

            # decide direção usando o vetor movimento projetado na normal
            mvx, mvy = (p_now[0]-p_prev[0], p_now[1]-p_prev[1])
            dir_sign = 1 if (mvx * nx + mvy * ny) > 0 else -1
            # Convenção:
            # dir_sign > 0 => IN ; dir_sign < 0 => OUT
            if dir_sign > 0:
                total_in += 1; crossed_in += 1
            else:
                total_out += 1; crossed_out += 1

            trk.last_cross_frame = trk.alive_frames

        timeline.append({
            "frame": frame_idx,
            "timeSec": round(ts, 3),
            "detections": len(dets),
            "crossedIn": crossed_in,
            "crossedOut": crossed_out,
            "cumulativeIn": total_in,
            "cumulativeOut": total_out
        })

        # ---------- Desenho ----------
        if writer is not None:
            draw_thick_line(frame, (x1, y1), (x2, y2), thickness=8, color=(0,255,255))
            for tid, trk in tracks.items():
                x, y, w, h = trk.bbox
                p1 = (int(x), int(y)); p2 = (int(x+w), int(y+h))
                cx, cy = int(trk.centroid()[0]), int(trk.centroid()[1])
                cv2.rectangle(frame, p1, p2, (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 3, (255,255,255), -1)
                put_text(frame, f"ID {tid}", (p1[0], max(0, p1[1]-6)))
            put_text(frame, f"IN: {total_in}  OUT: {total_out}", (10, 34), scale=0.8)

            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "video": Path(video_path).name,
        "frameSize": {"width": width, "height": height},
        "sourceFps": float(src_fps),
        "sampleFps": float(sample_fps),
        "linePx": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "totals": {"in": int(total_in), "out": int(total_out)},
        "timeline": timeline,
        "annotated_video": annotated_path
    }

# =================== Classe pública ===================
class PeopleLineCounter:
    def __init__(
        self,
        line_norm=(0.1, 0.5, 0.9, 0.5),
        sample_fps=5.0,
        save_annotated=True,
        out_dir=Path("outputs"),
        chunk_seconds=60,
        workers=0,
    ):
        self.line_norm = line_norm
        self.sample_fps = float(sample_fps)
        self.save_annotated = bool(save_annotated)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_seconds = int(chunk_seconds)
        self.workers = int(workers)

    def process(self, video_path: str):
        if self.chunk_seconds and self.chunk_seconds > 0:
            return self._process_parallel(video_path)
        return process_segment(
            video_path, self.line_norm, self.sample_fps, self.save_annotated, self.out_dir, 0.0, ""
        )

    # ---------- Streaming por chunk (mantido) ----------
    def iter_parallel(self, video_path: str):
        base = Path(video_path).stem
        seg_dir = self.out_dir / f"{base}_segments"
        seg_dir.mkdir(exist_ok=True)

        segments = split_into_chunks(video_path, str(seg_dir), chunk_seconds=self.chunk_seconds)
        if not segments:
            raise RuntimeError("Segmentação falhou (sem segmentos).")

        offsets, durs, acc = [], [], 0.0
        for p in segments:
            d = probe_duration(p)
            durs.append(d)
            offsets.append(acc)
            acc += d

        max_workers = self.workers if self.workers > 0 else (os.cpu_count() or 2)
        results = [None] * len(segments)
        annotated_segments = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futmap = {}
            for i, (seg, off) in enumerate(zip(segments, offsets)):
                fut = ex.submit(
                    process_segment,
                    seg, self.line_norm, self.sample_fps, self.save_annotated,
                    self.out_dir, off, f"_seg{i:03d}"
                )
                futmap[fut] = i

            for fut in as_completed(futmap):
                idx = futmap[fut]
                r = fut.result()
                results[idx] = r
                if self.save_annotated and r.get("annotated_video"):
                    annotated_segments.append(r["annotated_video"])

                start_sec = offsets[idx]
                end_sec   = offsets[idx] + (durs[idx] or 0.0)
                yield {
                    "event": "chunk_result",
                    "index": idx,
                    "startSec": round(float(start_sec), 3),
                    "endSec": round(float(end_sec), 3),
                    "in": int(r["totals"]["in"]),
                    "out": int(r["totals"]["out"]),
                    "annotated": r.get("annotated_video"),
                }

        total_in  = sum(r["totals"]["in"]  for r in results)
        total_out = sum(r["totals"]["out"] for r in results)
        timeline = []
        for r in results:
            timeline.extend(r["timeline"])

        annotated_concat = None
        if self.save_annotated and annotated_segments:
            annotated_concat = str(self.out_dir / f"{base}_annotated_concat.mp4")
            concat_videos_mp4(annotated_segments, annotated_concat)

        first = results[0]
        yield {
            "event": "final",
            "video": Path(video_path).name,
            "frameSize": first["frameSize"],
            "sourceFps": first["sourceFps"],
            "sampleFps": float(self.sample_fps),
            "linePx": first["linePx"],
            "totals": {"in": int(total_in), "out": int(total_out)},
            "timeline": timeline,
            "annotated_video": annotated_concat,
            "annotated_segments": annotated_segments if self.save_annotated else [],
        }

    # ---------- Execução paralela sem streaming (mantido) ----------
    def _process_parallel(self, video_path: str):
        base = Path(video_path).stem
        seg_dir = self.out_dir / f"{base}_segments"
        seg_dir.mkdir(exist_ok=True)
        segments = split_into_chunks(video_path, str(seg_dir), chunk_seconds=self.chunk_seconds)
        if not segments:
            raise RuntimeError("Segmentação falhou (sem segmentos).")

        offsets, acc = [], 0.0
        for p in segments:
            offsets.append(acc)
            acc += probe_duration(p)

        max_workers = self.workers if self.workers > 0 else (os.cpu_count() or 2)
        results = [None] * len(segments)
        annotated_segments = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futmap = {}
            for i, (seg, off) in enumerate(zip(segments, offsets)):
                fut = ex.submit(
                    process_segment,
                    seg, self.line_norm, self.sample_fps, self.save_annotated,
                    self.out_dir, off, f"_seg{i:03d}"
                )
                futmap[fut] = i
            for fut in as_completed(futmap):
                idx = futmap[fut]
                r = fut.result()
                results[idx] = r
                if self.save_annotated and r.get("annotated_video"):
                    annotated_segments.append(r["annotated_video"])

        total_in  = sum(r["totals"]["in"]  for r in results)
        total_out = sum(r["totals"]["out"] for r in results)
        timeline = []
        for r in results:
            timeline.extend(r["timeline"])

        annotated_concat = None
        if self.save_annotated and annotated_segments:
            annotated_concat = str(self.out_dir / f"{base}_annotated_concat.mp4")
            concat_videos_mp4(annotated_segments, annotated_concat)

        first = results[0]
        return {
            "video": Path(video_path).name,
            "frameSize": first["frameSize"],
            "sourceFps": first["sourceFps"],
            "sampleFps": float(self.sample_fps),
            "linePx": first["linePx"],
            "totals": {"in": int(total_in), "out": int(total_out)},
            "timeline": timeline,
            "annotated_video": annotated_concat,
            "annotated_segments": annotated_segments if self.save_annotated else [],
        }

# =================== Execução por pasta (opcional) ===================
if __name__ == "__main__":
    import argparse, glob, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=[],
                        help="Arquivos .mp4; se vazio, busca video*.mp4 no diretório atual.")
    parser.add_argument("--line", nargs=4, type=float, default=[0.10,0.55,0.90,0.55],
                        help="Linha normalizada x1 y1 x2 y2 (0..1)")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--chunk", type=int, default=45)
    parser.add_argument("--save-annot", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    vids = args.videos or sorted(glob.glob("video*.mp4"))
    plc = PeopleLineCounter(
        line_norm=tuple(args.line),
        sample_fps=args.fps,
        save_annotated=args.save_annot,
        out_dir=Path(args.out),
        chunk_seconds=args.chunk,
        workers=args.workers
    )
    summary = {}
    for v in vids:
        print(f"[INFO] Processando: {v}")
        res = plc.process(v)
        summary[Path(v).name] = res["totals"]
        print(f"  -> IN {res['totals']['in']} | OUT {res['totals']['out']}")
        if res.get("annotated_video"):
            print(f"  -> anotado: {res['annotated_video']}")
    Path(args.out).mkdir(exist_ok=True, parents=True)
    with open(Path(args.out)/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] summary salvo em outputs/summary.json")
