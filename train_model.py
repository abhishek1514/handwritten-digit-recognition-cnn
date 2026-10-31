
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import DigitCNN


# ==========================================
# STEP 1: DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "digit_cnn.pth"


# ==========================================
# STEP 2: IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==========================================
# STEP 3: LOAD MNIST DATASET
# ==========================================

train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# STEP 4: CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# ==========================================
# STEP 5: VERIFY DATASET
# ==========================================

print("\nDataset Information")

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Image tensor shape:", images[0].shape)
print("First few labels:", labels[:10].tolist())


# ==========================================
# STEP 6: VISUALIZE SAMPLE IMAGES
# ==========================================

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    img = images[i].squeeze() * 0.3081 + 0.1307

    ax.imshow(img.clamp(0, 1), cmap="gray")

    ax.set_title(f"Label: {labels[i].item()}")

    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# STEP 7: INITIALIZE CNN MODEL
# ==========================================

model = DigitCNN().to(device)

print("\nCNN Architecture:")
print(model)


# ==========================================
# STEP 8: LOSS FUNCTION AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# STEP 9: TRAIN CNN MODEL
# ==========================================

epochs = 5

print("\nStarting CNN Training...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# STEP 10: MODEL EVALUATION
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()


accuracy = 100.0 * correct / total

print("\nModel Evaluation")

print(f"Test Accuracy: {accuracy:.2f}%")


# ==========================================
# STEP 11: SAVE TRAINED MODEL
# ==========================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nModel saved successfully:", MODEL_PATH)
