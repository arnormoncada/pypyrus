"""
Minimal MLP training loop on MNIST.

Usage:  python examples/train_simple_mlp.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Config ---
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---
transform = transforms.ToTensor()
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.layers(x)
    
model = SimpleMLP().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# --- Train ---
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}/{EPOCHS}  loss: {total_loss / len(train_loader):.4f}")

print("Done.")
