import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

CAM_INDEX = 0
CLASS_NAME = "bottle"  
OBJ_WIDTH_M = 0.045   
DIST_M = 0.15           
CONF = 0.35
DEVICE = 0

model = YOLO("yolov8n.pt")
names = model.names
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

print("ENTER: capture+detect once")
print("c + ENTER: compute focal_px and save focal_px.txt")
print("q + ENTER: quit")

best_wpx = None

while True:
    cmd = input("> ").strip().lower()
    if cmd == "q":
        break

    ok, frame = cap.read()
    if not ok or frame is None:
        print("camera read failed")
        continue

    res = model.predict(frame, device=DEVICE, conf=CONF, verbose=False)[0]
    best = None
    best_conf = -1.0

    if res.boxes is not None:
        for b in res.boxes:
            cls_id = int(b.cls.item())
            cls = names.get(cls_id, str(cls_id))
            conf = float(b.conf.item())
            if cls == CLASS_NAME and conf > best_conf:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                best = (x1, x2, conf)
                best_conf = conf

    if best is None:
        print(f"no {CLASS_NAME} detected, adjust pose/light and retry")
        continue

    x1, x2, conf = best
    w_px = max(1.0, x2 - x1)
    best_wpx = w_px
    print(f"detected {CLASS_NAME} conf={conf:.2f} w_px={w_px:.1f}")

    if cmd == "c":
        focal_px = (DIST_M * w_px) / OBJ_WIDTH_M
        with open("focal_px.txt", "w", encoding="utf-8") as f:
            f.write(str(focal_px))
        print(f"FOCAL_PX={focal_px:.2f} saved to focal_px.txt")
        break

cap.release()
