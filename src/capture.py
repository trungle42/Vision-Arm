import sys
import time
from pathlib import Path

import cv2

label = sys.argv[1]  # pen, cube or nothing
out = Path("data/raw") / label
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("could not open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # same size for training photos and the demo
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("camera size:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

count = len(list(out.glob("*.jpg")))
print(f"saving to {out}, {count} photos so far. space = save, q = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    shown = frame.copy()
    cv2.putText(shown, f"{label}: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("capture", shown)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(str(out / f"cam_{int(time.time() * 1000)}.jpg"), frame)
        count += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
