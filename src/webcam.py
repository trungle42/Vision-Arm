import json

import cv2
import torch
from PIL import Image
from torchvision import models, transforms

from arm import pipeline

CLASSES = json.load(open("models/classes.json"))

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("models/model.pth", map_location="cpu", weights_only=True))
model.eval()

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("could not open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # same size for training photos and the demo
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("camera size:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

H = pipeline.load_calibration()
background = None  # press b with an empty scene to set it

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, the model was trained on RGB
    inp = tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]

    best = int(probs.argmax())
    label = f"{CLASSES[best]} {probs[best]:.0%}"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # where is it? whatever differs from the empty reference frame
    if background is not None:
        diff = cv2.absdiff(background, gray)
        _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > 500:
                x, y, w, h = cv2.boundingRect(biggest)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx}, {cy})", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                result = pipeline.pixel_to_pulses(H, cx, cy)
                if result is None:
                    arm_text = "out of reach"
                else:
                    (tx, ty), degs, pulses = result
                    arm_text = f"table ({tx:.1f}, {ty:.1f}) cm  servos {pulses[0]}, {pulses[1]} us"
                cv2.putText(frame, arm_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    cv2.imshow("vision-arm", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("b"):
        background = gray.copy()
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
