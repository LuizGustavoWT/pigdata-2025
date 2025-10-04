import os
import subprocess
from pathlib import Path
from typing import List, Tuple

def run_ffmpeg(cmd: list):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc

def split_into_chunks(input_video: str, out_dir: str, chunk_seconds: int = 60) -> List[str]:
    """
    Corta o vídeo em segmentos de 'chunk_seconds' usando o segment muxer.
    - Reencode para H.264 + AAC para garantir concat posterior (parâmetros idênticos).
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    pattern = outp / "seg_%03d.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_seconds})",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        str(pattern)
    ]
    run_ffmpeg(cmd)

    segs = sorted([str(p) for p in outp.glob("seg_*.mp4")])
    if not segs:
        raise RuntimeError("Segmentação FFmpeg não gerou arquivos.")
    return segs

def concat_videos_mp4(segments: List[str], output_path: str):
    """
    Concatena segmentos mp4 com o demuxer 'concat' (sem recodificar).
    Exige todos com os mesmos codecs/parâmetros (garantimos acima).
    """
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    list_file = outp.parent / "_concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"file '{Path(s).as_posix()}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(outp)
    ]
    run_ffmpeg(cmd)
    try:
        list_file.unlink()
    except:
        pass
