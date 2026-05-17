import os
from typing import Any, Optional, Tuple

import numpy as np
import rerun as rr

_RR_STEP = 0

try:
    import cv2  
except Exception:
    cv2 = None  


try:
    from ultralytics import YOLO  
except Exception:
    YOLO = None 

_YOLO_MODEL: Optional["YOLO"] = None


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _get_yolo_model() -> Optional["YOLO"]:
    """
    Lazy-load YOLO model once.
    Configure weights via env var:
      LEROBOT_YOLO_WEIGHTS = path/to/weights.pt
    """
    global _YOLO_MODEL
    if YOLO is None:
        return None
    if _YOLO_MODEL is None:
        weights = os.getenv("LEROBOT_YOLO_WEIGHTS", "yolov8n.pt")
        _YOLO_MODEL = YOLO(weights)
    return _YOLO_MODEL


def _is_image_array(val: np.ndarray) -> bool:
    return isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[2] in (3, 4)


def _estimate_distance_m_from_width_px(w_px: float) -> Optional[float]:
    """
    Single-object monocular ranging:
      distance = fx * W_real / w_px

    Your calibration:
      W_real = 0.05 m
      fx ≈ 545 px  (from D0=0.25m, W0px=109px)
      fx * W_real = 27.25
      distance_m = 27.25 / w_px
    """
    if w_px <= 1.0:
        return None
    d = 27.25 / w_px
    if not (0.05 <= d <= 2.5):
        return None
    return float(d)


def _to_uint8_hwc_bgr(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Ensure image is HxWx3 uint8 in BGR for YOLO + cv2 drawing.
    Returns None if unsupported.
    """
    if not isinstance(img, np.ndarray):
        return None
    if img.ndim != 3:
        return None

    # If CHW, convert to HWC
    if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))

    # Drop alpha
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if img.shape[2] != 3:
        return None

    # Convert dtype/range
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        # assume 0..1 or 0..255
        mx = float(np.nanmax(img)) if img.size else 0.0
        if mx <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    try:
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    except Exception:
        return None


def _yolo_best_box(img_bgr_uint8: np.ndarray) -> Optional[Tuple[float, float, float, float, float, int]]:
    """
    Returns (x1,y1,x2,y2,conf,cls_id) for best detection or None.
    """
    model = _get_yolo_model()
    if model is None:
        return None

    res = model(img_bgr_uint8, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None

    best_i = None
    best_conf = -1.0
    for i in range(len(res.boxes)):
        conf = float(res.boxes[i].conf[0].cpu().numpy())
        if conf > best_conf:
            best_conf = conf
            best_i = i
    if best_i is None:
        return None

    b = res.boxes[best_i].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(float, b.tolist())
    cls_id = int(res.boxes[best_i].cls[0].cpu().numpy())
    return (x1, y1, x2, y2, float(best_conf), cls_id)


def _annotate_image_with_yolo_and_range(img: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return img

    img_bgr = _to_uint8_hwc_bgr(img)
    if img_bgr is None:
        return img

    det = _yolo_best_box(img_bgr)
    if det is None:
        return img_bgr

    x1, y1, x2, y2, conf, cls_id = det
    w_px = x2 - x1
    dist_m = _estimate_distance_m_from_width_px(w_px)

    model = _get_yolo_model()
    if model is not None:
        try:
            cls_name = model.names.get(cls_id, str(cls_id)) 
        except Exception:
            cls_name = str(cls_id)
    else:
        cls_name = str(cls_id)

    if dist_m is None:
        label = f"{cls_name} conf={conf:.2f}"
    else:
        label = f"{cls_name} {dist_m:.2f}m conf={conf:.2f}"

    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(img_bgr, p1, p2, (0, 255, 0), 2)

    tx = p1[0]
    ty = max(0, p1[1] - 8)
    cv2.putText(img_bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_bgr


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    global _RR_STEP
    rr.set_time_sequence("step", _RR_STEP)
    _RR_STEP += 1

    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalar(float(v)))
            else:
                entity = f"observation.{obs}"
                obs_lower = str(obs).lower()

                if _is_image_array(val) and ("front" in obs_lower):
                    annotated = _annotate_image_with_yolo_and_range(val)
                    rr.log(entity, rr.Image(annotated))
                else:
                    rr.log(entity, rr.Image(val))

    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalar(float(v)))
