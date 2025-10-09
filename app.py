# app.py — BusSight • Flask entrypoint (render, upload, process, SSE, download, summary, cleanup)
import os

# ======= Forçar CPU / evitar CUDA =======
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Desabilita GPU
os.environ["ULTRALYTICS_FORCE_CPU"] = "1"        # Força CPU no Ultralytics
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Evita MPS em Macs sem suporte
os.environ.setdefault("OMP_NUM_THREADS", "4")    # (opcional) limita threads

import io, json, time, uuid, math, mimetypes, traceback
from datetime import datetime, timedelta
from urllib.parse import quote
from flask import Flask, render_template, request, jsonify, Response, send_file, abort

from werkzeug.utils import secure_filename

# wrappers com o contrato exigido
# (OBS: yolo_counter também deve respeitar device='cpu' e half=False, ver nota abaixo)
from yolo_counter import process_video, process_stream

# Configuração básica
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXT = {".mp4", ".avi", ".mkv", ".mov"}

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__)

def _cleanup_old_files(hours=48):
    """Remove arquivos com mais de 'hours' horas em uploads/ e outputs/."""
    cutoff = time.time() - hours * 3600
    for dirp in (UPLOADS_DIR, OUTPUTS_DIR):
        for root, _, files in os.walk(dirp):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                except Exception:
                    pass

_cleanup_old_files()

@app.route("/")
def index():
    # Renderiza templates/index.html
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Recebe o vídeo, valida extensão e grava em uploads/ com nome único."""
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "Arquivo 'video' não encontrado."}), 400
    f = request.files["video"]
    if not f or f.filename == "":
        return jsonify({"ok": False, "error": "Nenhum arquivo selecionado."}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "Extensão inválida. Use mp4/avi/mkv/mov."}), 400

    uniq = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOADS_DIR, uniq)
    f.save(path)
    rel_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")
    return jsonify({"ok": True, "video_path": rel_path})

@app.route("/process", methods=["POST"])
def process_batch():
    """Processamento em lote: chama process_video e devolve payload completo."""
    try:
        data = request.get_json(force=True)
        video_path = data.get("video_path")
        if not video_path:
            return jsonify({"ok": False, "error": "video_path ausente."}), 400

        # Normaliza caminho relativo
        abs_video = os.path.abspath(os.path.join(BASE_DIR, video_path))
        if not abs_video.startswith(UPLOADS_DIR):
            return jsonify({"ok": False, "error": "Caminho inválido."}), 400
        if not os.path.exists(abs_video):
            return jsonify({"ok": False, "error": "Arquivo não encontrado."}), 404

        line = data.get("line", [])
        if not (isinstance(line, list) and len(line) == 4):
            return jsonify({"ok": False, "error": "Linha inválida (x1,y1,x2,y2)."}), 400

        sample_fps = float(data.get("sample_fps", 5.0))
        chunk_seconds = int(data.get("chunk_seconds", 60))
        workers = int(data.get("workers", 0))
        save_annotated = bool(data.get("save_annotated", True))

        payload = process_video(
            abs_video,
            line_norm=tuple(map(float, line)),
            sample_fps=sample_fps,
            chunk_seconds=chunk_seconds,
            workers=workers,
            save_annotated=save_annotated,
            # se o wrapper aceitar kwargs extras, garanta CPU
            device="cpu",
            half=False,
        )
        payload["ok"] = True
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("Erro em /process")
        # Mensagem curta para a UI (sem stacktrace gigante)
        return jsonify({"ok": False, "error": _short_error(e)}), 500

def _sse_format(d):
    """Formata dict -> linha SSE 'data:'."""
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

@app.route("/process/stream")
def process_stream_endpoint():
    """SSE: emite 'progress' e finaliza com 'done' com o mesmo payload do /process."""
    try:
        video_path = request.args.get("video_path", "")
        if not video_path:
            return jsonify({"ok": False, "error": "video_path ausente."}), 400

        abs_video = os.path.abspath(os.path.join(BASE_DIR, video_path))
        if not abs_video.startswith(UPLOADS_DIR):
            return jsonify({"ok": False, "error": "Caminho inválido."}), 400
        if not os.path.exists(abs_video):
            return jsonify({"ok": False, "error": "Arquivo não encontrado."}), 404

        x1 = float(request.args.get("x1", "0"))
        y1 = float(request.args.get("y1", "0"))
        x2 = float(request.args.get("x2", "1"))
        y2 = float(request.args.get("y2", "1"))
        sample_fps = float(request.args.get("sample_fps", "5.0"))
        chunk_seconds = int(request.args.get("chunk_seconds", "60"))
        workers = int(request.args.get("workers", "0"))

        def generate():
            try:
                for ev in process_stream(
                    abs_video,
                    line_norm=(x1, y1, x2, y2),
                    sample_fps=sample_fps,
                    chunk_seconds=chunk_seconds,
                    workers=workers,
                    save_annotated=False,
                    device="cpu",
                    half=False,
                ):
                    yield _sse_format(ev)
            except Exception as e:
                yield _sse_format({"type": "error", "message": _short_error(e)})

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        app.logger.exception("Erro em /process/stream")
        return jsonify({"ok": False, "error": _short_error(e)}), 500

@app.route("/download/<path:filepath>")
def download(filepath):
    """Serve APENAS arquivos dentro de outputs/ (proteção contra path traversal)."""
    # Normaliza para outputs/
    abs_path = os.path.abspath(os.path.join(OUTPUTS_DIR, filepath))
    if not abs_path.startswith(OUTPUTS_DIR):
        abort(403)
    if not os.path.exists(abs_path):
        abort(404)
    mime, _ = mimetypes.guess_type(abs_path)
    return send_file(abs_path, as_attachment=True, mimetype=mime)

@app.route("/report/summary")
def report_summary():
    """Resumo rápido se resultados já existem (opcional)."""
    video_path = request.args.get("video_path", "")
    if not video_path:
        return jsonify({"ok": False, "error": "video_path ausente."}), 400
    abs_video = os.path.abspath(os.path.join(BASE_DIR, video_path))
    if not abs_video.startswith(UPLOADS_DIR) or not os.path.exists(abs_video):
        return jsonify({"ok": False, "error": "Vídeo inválido ou não encontrado."}), 400

    # Convenção: procura por CSV com mesmo stem no outputs/
    stem = os.path.splitext(os.path.basename(abs_video))[0]
    csv_guess = os.path.join(OUTPUTS_DIR, f"{stem}_windows.csv")
    if os.path.exists(csv_guess):
        return jsonify({"ok": True, "csv_path": os.path.relpath(csv_guess, BASE_DIR).replace('\\','/')})
    return jsonify({"ok": False, "error": "Resumo indisponível."}), 404

# ======= Tratador global de erros (mensagem curta na UI) =======
@app.errorhandler(Exception)
def _handle_any_error(e):
    app.logger.exception("Erro não tratado")
    return jsonify({"ok": False, "error": _short_error(e)}), 500

def _short_error(e: Exception) -> str:
    msg = str(e) or e.__class__.__name__
    # mensagens enormes de CUDA -> mensagem curta
    if "CUDA" in msg or "kernel image is not available" in msg:
        return "Falha de GPU/CUDA detectada. Rodando em CPU. Verifique versões apenas se quiser GPU."
    return msg[:300]  # evita resposta gigante

if __name__ == "__main__":
    # Dev: python app.py
    # Prod (opcional): waitress-serve --host 0.0.0.0 --port 8000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
