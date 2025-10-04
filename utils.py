import cv2
import numpy as np

def iou_xywh(b1, b2):
    # b = [x,y,w,h]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return 0.0 if union <= 0 else inter / union

def point_side_of_line(px, py, x1, y1, x2, y2):
    # sinal do produto vetorial
    v = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if v > 0 else (-1 if v < 0 else 0)

def put_text(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default
