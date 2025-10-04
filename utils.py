# utils.py
import cv2
import math
from collections import deque

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

def put_text(img, text, org, scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ======= ADIÇÕES =======

def draw_thick_line(img, p1, p2, thickness=8, color=(0,255,255)):
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def signed_distance_to_line(px, py, x1, y1, x2, y2):
    # distância assinada da reta (neg/pos conforme lado)
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    denom = math.hypot(A, B)
    if denom == 0:
        return 0.0
    return (A * px + B * py + C) / denom

def segments_intersect(p1, p2, q1, q2):
    # interseção entre segmentos p1->p2 e q1->q2
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on_seg(a, b, c):
        return (min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and
                min(a[1],b[1]) <= c[1] <= max(a[1],b[1]))
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    if (o1 == 0 and on_seg(p1, p2, q1)) or \
       (o2 == 0 and on_seg(p1, p2, q2)) or \
       (o3 == 0 and on_seg(q1, q2, p1)) or \
       (o4 == 0 and on_seg(q1, q2, p2)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

class Track:
    __slots__ = ("id","bbox","misses","history","last_side","last_cross_frame","alive_frames")
    def __init__(self, tid, bbox, side, max_hist=8):
        self.id = tid
        self.bbox = bbox
        self.misses = 0
        self.history = deque(maxlen=max_hist)
        cx = bbox[0] + bbox[2]/2.0
        cy = bbox[1] + bbox[3]/2.0
        self.history.append((cx, cy))
        self.last_side = side
        self.last_cross_frame = -999999
        self.alive_frames = 0

    def update(self, bbox):
        self.bbox = bbox
        cx = bbox[0] + bbox[2]/2.0
        cy = bbox[1] + bbox[3]/2.0
