# cpu_tuning.py
import os
import cv2
import torch

def tune_cpu_threads(num_infer_threads: int = 4, num_interop_threads: int = 1, opencv_threads: int = 0):
    """
    num_infer_threads: threads de computação (MKL/OMP) que o PyTorch usa
    num_interop_threads: paralelismo interop (geralmente 1 em CPU)
    opencv_threads: 0 = deixe OpenCV decidir; 1 = desabilita paralelismo do OpenCV
    """
    # variáveis de ambiente para MKL/OMP (devem ser setadas antes de carregar libs)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_infer_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_infer_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_infer_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_infer_threads))

    try:
        torch.set_num_threads(num_infer_threads)
        torch.set_num_interop_threads(num_interop_threads)
    except Exception:
        pass

    try:
        if opencv_threads == 1:
            cv2.setNumThreads(1)  # evita sobreposição com MKL/OMP
    except Exception:
        pass