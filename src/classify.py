import torch
from torchvision import models

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/model.pth", map_location="cpu", weights_only=True))
model.eval()
print("loaded ok")

from PIL import Image
from torchvision import transforms
import glob

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
path = glob.glob("data/raw/pen/*")[0]
x = tf(Image.open(path).convert("RGB")).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
print(["cube", "pen"], probs)
