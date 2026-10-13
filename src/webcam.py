import json

import cv2
import torch
from PIL import Image
from torchvision import models, transforms

CLASSES = json.load(open("models/classes.json"))

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("models/model.pth", map_location="cpu", weights_only=True))
model.eval()

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, the model was trained on RGB
    x = tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    best = int(probs.argmax())
    label = f"{CLASSES[best]} {probs[best]:.0%}"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("vision-arm", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
