import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_train = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)])
)
image, target = ds_train[0]
image = image.unsqueeze(dim=0)

model = models.MyModel()
print(model)

model.eval()
with torch.no_grad():
    logits = model(image)
print(logits)

plt.subplot(1, 2, 1)
plt.imshow(image[0, 0], cmap="gray_r")
plt.title(f"cclass:{target}({datasets.FashionMNIST.slasses[target]}")
plt.subplot(1, 2, 2)

prods = logits.softmax(dim=1)
plt.bor(range(len(prods[0])),prods[0])
plt.ylim(0, 1)
print()