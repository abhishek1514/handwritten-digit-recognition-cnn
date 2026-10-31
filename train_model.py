
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ==========================================
# PHASE 1: DATA ACQUISITION & PREPROCESSING
# ==========================================

# STEP 1: Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# STEP 2: Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# STEP 3: Download MNIST training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# STEP 4: Download MNIST testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# STEP 5: Create DataLoaders
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


# STEP 6: Verify dataset
print("\nDataset Information")

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Image tensor shape:", images[0].shape)
print("First few labels:", labels[:10].tolist())


# STEP 7: Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    # Reverse normalization for visualization
    img = images[i].squeeze() * 0.3081 + 0.1307

    ax.imshow(img.clamp(0, 1), cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# PHASE 2: CNN MODEL DEVELOPMENT & TRAINING
# ==========================================

# STEP 8: Define CNN architecture
class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction layers
        self.conv_layers = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # Classification layers
        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x


# STEP 9: Initialize model
model = DigitCNN().to(device)

print("\nCNN Architecture:")
print(model)


# STEP 10: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# STEP 11: Train the model
epochs = 5

print("\nStarting CNN Training...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# STEP 12: Evaluate model on test dataset
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


# STEP 13: Save trained model weights
torch.save(
    model.state_dict(),
    "digit_cnn.pth"
)

print("\nModel saved successfully: digit_cnn.pth")
