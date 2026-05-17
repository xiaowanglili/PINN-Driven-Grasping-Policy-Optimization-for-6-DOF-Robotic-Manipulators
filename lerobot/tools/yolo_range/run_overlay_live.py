# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

CAM_INDEX = 0
CONF = 0.35
IOU = 0.50
DEVICE = 0

KNOWN_WIDTH_M = {
    "bottle": 0.075,
    "book": 0.150,
    "cell phone": 0.070,
}

FOCAL_PX = 800.0
try:
    with open("focal_px.txt", "r") as f:
        FOCAL_PX = float(f.read().strip())
except:
    pass

def estimate_distance(f_px, obj_width_m, bbox_w_px):
    if f_px <= 0 or obj_width_m <= 0 or bbox_w_px <= 1:
        return None
    return (f_px * obj_width_m) / bbox_w_px

def main():
    model = YOLO("yolov8n.pt")
    names = model.names

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    plt.ion()
    fig, ax = plt.subplots()
    im = None

    last_t = time.time()
    frames = 0
    fps = 0.0

    print("Press Ctrl+C to exit")
    print("Using focal_px =", FOCAL_PX)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            res = model.predict(frame, device=DEVICE, conf=CONF, iou=IOU, verbose=False)[0]

            if res.boxes is not None:
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    cls = names.get(cls_id, str(cls_id))
                    conf = float(b.conf.item())

                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    w_px = max(1.0, x2 - x1)
                    label = f"{cls} {conf:.2f}"

                    if cls in KNOWN_WIDTH_M:
                        dist = estimate_distance(FOCAL_PX, KNOWN_WIDTH_M[cls], w_px)
                        if dist is not None:
                            label += f" | {dist:.2f} m"

                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
                    cv2.putText(frame, label, (x1i, max(30, y1i-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            frames += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = frames / (now - last_t)
                frames = 0
                last_t = now

            cv2.putText(frame, f"FPS {fps:.1f}", (10,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if im is None:
                im = ax.imshow(rgb)
                ax.axis("off")
            else:
                im.set_data(rgb)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass

    cap.release()
    plt.ioff()

if __name__ == "__main__":
    main()
