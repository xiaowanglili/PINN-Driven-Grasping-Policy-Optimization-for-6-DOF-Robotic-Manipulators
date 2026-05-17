import cv2

def test(backend_name, backend, max_idx=6):
    print("\n==", backend_name, "==")
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx, backend)
        ok = cap.isOpened()
        print(f"idx={idx} opened={ok}")
        if ok:
            ret, frame = cap.read()
            print("  read=", ret, "shape=", None if frame is None else frame.shape)
        cap.release()

test("MSMF", cv2.CAP_MSMF)
test("DSHOW", cv2.CAP_DSHOW)
