
import torch.nn as nn


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
