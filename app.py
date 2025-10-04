import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import json
from yolo_counter import PeopleLineCounter

app = FastAPI(title="YOLO People Counter")
BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "uploads"
OUTPUTS = BASE_DIR / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    sample_fps: float = Form(5.0),
    save_annotated: int = Form(1),
    chunk_seconds: int = Form(60),
    workers: int = Form(0),
):
    # validações
    for v in (x1, y1, x2, y2):
        if not (0.0 <= v <= 1.0):
            return JSONResponse({"error": "x1,y1,x2,y2 devem estar entre 0 e 1."}, status_code=400)
    sample_fps = max(0.5, float(sample_fps))
    chunk_seconds = max(10, int(chunk_seconds))
    workers = max(0, int(workers))

    # salva o vídeo
    dst_path = UPLOADS / video.filename
    with dst_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    # cria o contador já com configs de chunk e paralelismo
    counter = PeopleLineCounter(
        line_norm=(x1, y1, x2, y2),
        sample_fps=sample_fps,
        save_annotated=bool(save_annotated),
        out_dir=OUTPUTS,
        chunk_seconds=chunk_seconds,
        workers=workers,
    )

    try:
        result = counter.process(str(dst_path))
    except Exception as e:
        return JSONResponse({"error": "Falha ao processar vídeo", "detail": str(e)}, status_code=500)

    # links de download
    if result.get("annotated_video"):
        result["annotated_video_url"] = f"/download/{Path(result['annotated_video']).name}"
    if result.get("annotated_segments"):
        result["annotated_segment_urls"] = [
            f"/download/{Path(p).name}" for p in result["annotated_segments"]
        ]

    return JSONResponse(result)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUTS / filename
    if not file_path.exists():
        return JSONResponse({"error": "Arquivo não encontrado"}, status_code=404)
    return FileResponse(str(file_path), media_type="application/octet-stream", filename=filename)

@app.post("/upload_stream")
async def upload_video_stream(
    request: Request,
    video: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    sample_fps: float = Form(5.0),
    save_annotated: int = Form(1),
    chunk_seconds: int = Form(60),
    workers: int = Form(0),
    window_minutes: int = Form(5),         # tamanho da faixa (ex.: 5 min)
    clock_start: str = Form("")            # opcional "HH:MM" p/ rótulos tipo 14:20-14:25
):
    for v in (x1, y1, x2, y2):
        if not (0.0 <= v <= 1.0):
            return JSONResponse({"error": "x1,y1,x2,y2 devem estar entre 0 e 1."}, status_code=400)
    sample_fps = max(0.5, float(sample_fps))
    chunk_seconds = max(10, int(chunk_seconds))
    workers = max(0, int(workers))
    window_minutes = max(1, int(window_minutes))

    # salva o vídeo
    dst_path = UPLOADS / video.filename
    with dst_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    counter = PeopleLineCounter(
        line_norm=(x1, y1, x2, y2),
        sample_fps=sample_fps,
        save_annotated=bool(save_annotated),
        out_dir=OUTPUTS,
        chunk_seconds=chunk_seconds,
        workers=workers,
    )

    async def gen():
        try:
            # envia um header inicial
            yield json.dumps({"event":"start","message":"processing"}) + "\n"

            final_payload = None
            # itera chunks conforme vão terminando
            for ev in counter.iter_parallel(str(dst_path)):
                if ev["event"] == "chunk_result":
                    # envia parcial
                    yield json.dumps(ev) + "\n"
                elif ev["event"] == "final":
                    # agrega por faixas de horário
                    windows = PeopleLineCounter.aggregate_windows(ev["timeline"], window_minutes, clock_start)
                    ev["windows"] = windows
                    # monta URLs de download
                    if ev.get("annotated_video"):
                        ev["annotated_video_url"] = f"/download/{Path(ev['annotated_video']).name}"
                    if ev.get("annotated_segments"):
                        ev["annotated_segment_urls"] = [
                            f"/download/{Path(p).name}" for p in ev["annotated_segments"]
                        ]
                    final_payload = ev
            # envia final
            if final_payload:
                yield json.dumps(final_payload) + "\n"
        except Exception as e:
            yield json.dumps({"event":"error","detail":str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="text/plain")