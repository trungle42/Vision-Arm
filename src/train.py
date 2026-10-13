import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
data = datasets.ImageFolder("data/raw", transform=tfm)  # labels come from the folder names
print(data.classes, len(data))

n_test = int(0.2 * len(data))
g = torch.Generator().manual_seed(42)
train_set, test_set = random_split(data, [len(data) - n_test, n_test], generator=g)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    print(f"epoch {epoch+1}: loss {total_loss/len(train_set):.3f}, train acc {correct/len(train_set):.2f}")

model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
print(f"test acc: {correct}/{len(test_set)} = {correct/len(test_set):.2f}")

torch.save(model.cpu().state_dict(), "models/model.pth")
json.dump(data.classes, open("models/classes.json", "w"))
