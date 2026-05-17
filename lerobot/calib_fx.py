import cv2
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"

CAM_INDEX = 0

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

print("Press ESC to quit. Put the bottle at the reference distance (e.g. 0.50m).")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model(frame, verbose=False)[0]

    w_px = None
    best = None
    if res.boxes is not None and len(res.boxes) > 0:
        best_i = None
        best_conf = -1.0
        for i in range(len(res.boxes)):
            conf = float(res.boxes[i].conf[0].cpu().numpy())
            if conf > best_conf:
                best_conf = conf
                best_i = i
        if best_i is not None:
            b = res.boxes[best_i].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, b.tolist())
            w_px = float(x2 - x1)
            best = (x1, y1, x2, y2, best_conf)

    if best is not None:
        x1, y1, x2, y2, conf = best
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"bbox_w={w_px:.1f}px conf={conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        print(f"bbox_width_px={w_px:.1f}")

    cv2.imshow("fx_calib_front", frame)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
