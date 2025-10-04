import subprocess
from pathlib import Path

def _run(cmd: list):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc

def probe_duration(path: str) -> float:
    """Retorna duração (s) com ffprobe; se falhar, 0.0."""
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "format=duration","-of","default=nw=1:nk=1", str(path)
    ]
    try:
        out = _run(cmd).stdout.decode().strip()
        return float(out)
    except Exception:
        return 0.0

def split_into_chunks(input_video: str, out_dir: str, chunk_seconds: int = 60):
    """
    Corta em segmentos uniformes reencodados (H.264 + AAC).
    Reencodar garante concat 'sem recode' depois (parâmetros idênticos).
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    pattern = outp / "seg_%03d.mp4"

    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", input_video,
        "-c:v","libx264","-preset","veryfast","-crf","23",
        "-c:a","aac","-ar","48000","-ac","2",
        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_seconds})",
        "-f","segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps","1",
        str(pattern)
    ]
    _run(cmd)
    segs = sorted([str(p) for p in outp.glob("seg_*.mp4")])
    if not segs:
        raise RuntimeError("Segmentação FFmpeg não gerou arquivos.")
    return segs

def concat_videos_mp4(segments, output_path: str):
    """Concatena MP4s com demuxer concat (sem recodificar)."""
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    list_file = outp.parent / "_concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"file '{Path(s).as_posix()}'\n")
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","concat","-safe","0","-i", str(list_file),
        "-c","copy", str(outp)
    ]
    _run(cmd)
    try: list_file.unlink()
    except: pass
