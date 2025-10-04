import cv2
from ultralytics import YOLO
from utils import iou_xywh, point_side_of_line, put_text
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils_ffmpeg import split_into_chunks, concat_videos_mp4, probe_duration

def process_segment(video_path, line_norm, sample_fps, save_annotated, out_dir, time_offset=0.0, suffix=""):
    """Processa UM arquivo (vídeo inteiro OU segmento) – sequência básica + vídeo anotado opcional."""
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # linha px
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

    next_id = 1
    tracks = {}
    max_misses = 10
    iou_th = 0.3
    total_in = total_out = 0
    timeline = []
    frame_idx = -1
    out_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        out_idx += 1
        ts = out_idx / float(sample_fps) + float(time_offset)

        results = model(frame, classes=[0], conf=0.4, iou=0.5, verbose=False)
        dets = []
        if len(results):
            b = results[0].boxes
            if b is not None and b.xywh is not None:
                xywh = b.xywh.cpu().numpy()
                scores = b.conf.cpu().numpy()
                for (cx, cy, w, h), sc in zip(xywh, scores):
                    if sc < 0.4:
                        continue
                    x = float(cx - w/2); y = float(cy - h/2)
                    dets.append([x, y, float(w), float(h)])

        # associação por IoU
        for tid in list(tracks.keys()):
            tracks[tid]["updated"] = False
        for db in dets:
            best_id, best_iou = None, 0.0
            for tid, trk in tracks.items():
                score = iou_xywh(db, trk["bbox"])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= iou_th:
                trk = tracks[best_id]
                trk["bbox"] = db
                trk["centroid"] = (db[0]+db[2]/2.0, db[1]+db[3]/2.0)
                trk["misses"] = 0
                trk["updated"] = True
            else:
                tracks[next_id] = {
                    "bbox": db,
                    "centroid": (db[0]+db[2]/2.0, db[1]+db[3]/2.0),
                    "misses": 0,
                    "updated": True,
                    "last_side": None
                }
                next_id += 1

        # aging/remover
        for tid in list(tracks.keys()):
            if not tracks[tid]["updated"]:
                tracks[tid]["misses"] += 1
            if tracks[tid]["misses"] > max_misses:
                del tracks[tid]

        crossed_in = 0
        crossed_out = 0
        for tid, trk in tracks.items():
            cx, cy = trk["centroid"]
            side = point_side_of_line(cx, cy, x1, y1, x2, y2)
            last = trk.get("last_side", None)
            if last is None:
                trk["last_side"] = side
            else:
                if side != 0 and last != 0 and side != last:
                    if last < side:
                        total_in += 1; crossed_in += 1
                    else:
                        total_out += 1; crossed_out += 1
                    trk["last_side"] = side
                else:
                    trk["last_side"] = side

        timeline.append({
            "frame": frame_idx,
            "timeSec": round(ts, 3),
            "detections": len(dets),
            "crossedIn": crossed_in,
            "crossedOut": crossed_out,
            "cumulativeIn": total_in,
            "cumulativeOut": total_out
        })

        if writer is not None:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for tid, trk in tracks.items():
                x, y, w, h = trk["bbox"]
                p1 = (int(x), int(y)); p2 = (int(x+w), int(y+h))
                cx, cy = int(trk["centroid"][0]), int(trk["centroid"][1])
                cv2.rectangle(frame, p1, p2, (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 3, (255,255,255), -1)
                put_text(frame, f"ID {tid}", (p1[0], max(0, p1[1]-6)))
            put_text(frame, f"IN: {total_in}  OUT: {total_out}", (10, 30))
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
        self.out_dir.mkdir(exist_ok=True)
        self.chunk_seconds = int(chunk_seconds)
        self.workers = int(workers)

    def process(self, video_path: str):
        """Se chunk_seconds > 0: paralelo; senão, sequencial."""
        if self.chunk_seconds and self.chunk_seconds > 0:
            return self._process_parallel(video_path)
        return process_segment(
            video_path, self.line_norm, self.sample_fps, self.save_annotated, self.out_dir, 0.0, ""
        )

    # ========= NOVO: streaming de parciais por chunk =========
    def iter_parallel(self, video_path: str):
        """
        Generator que:
          - divide em chunks,
          - processa em paralelo,
          - YIELD por chunk:
              {"event":"chunk_result","index":i,"startSec":s0,"endSec":s1,"in":X,"out":Y,"annotated":path?}
          - ao final, YIELD:
              {"event":"final", ... agregado ...}
        """
        base = Path(video_path).stem
        seg_dir = self.out_dir / f"{base}_segments"
        seg_dir.mkdir(exist_ok=True)

        # 1) split
        segments = split_into_chunks(video_path, str(seg_dir), chunk_seconds=self.chunk_seconds)
        if not segments:
            raise RuntimeError("Segmentação falhou (sem segmentos).")

        # offsets/durações reais
        offsets, durs, acc = [], [], 0.0
        for p in segments:
            d = probe_duration(p)
            durs.append(d)
            offsets.append(acc)
            acc += d

        # 2) paralelo
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

                # parcial do chunk
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

        # 3) agregado final
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

    # ========= NOVO: agregação por faixas (14:20–14:25 etc.) =========
    @staticmethod
    def aggregate_windows(timeline, window_minutes=5, clock_start=""):
        """
        timeline: lista com 'timeSec','crossedIn','crossedOut'
        window_minutes: tamanho do bucket em minutos
        clock_start: "HH:MM" (opcional) p/ rótulo por relógio; se vazio, usa mm:ss.
        """
        bucket = max(1, int(window_minutes)) * 60
        if not timeline:
            return []

        # (t, in, out) em segundos inteiros
        events = [(int(round(x["timeSec"])), int(x["crossedIn"]), int(x["crossedOut"])) for x in timeline]
        max_t = max(t for t, _, _ in events)

        base_h = base_m = 0
        if clock_start:
            try:
                base_h, base_m = [int(x) for x in clock_start.split(":")]
            except:
                base_h = base_m = 0

        def to_clock(sec):
            total_min = sec // 60
            h = (base_h + (base_m + total_min) // 60) % 24
            m = (base_m + total_min) % 60
            return f"{h:02d}:{m:02d}"

        def mmss(sec):
            m = sec // 60; s = sec % 60
            return f"{m:02d}:{s:02d}"

        windows = []
        for start in range(0, max_t + 1, bucket):
            end = start + bucket
            win_in = win_out = 0
            for t, ci, co in events:
                if start <= t < end:
                    win_in += ci
                    win_out += co
            label = f"{to_clock(start)}-{to_clock(end)}" if clock_start else f"{mmss(start)}-{mmss(end)}"
            windows.append({"label": label, "in": win_in, "out": win_out})

        return windows

    # (continua igual) processamento completo sem stream (retorno único)
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