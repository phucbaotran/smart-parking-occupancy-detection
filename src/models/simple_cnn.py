# =========================================================
# File name: simple_cnn.py
# Project: Smart Parking Occupancy Detection
# Description: Simple CNN model for parking occupancy classification
# =========================================================


# *********************** Supporting libraries
import torch
import torch.nn as nn


# *********************** Supporting functions
def getNumberOfClasses():
    """
    Return the number of output classes.

    Class 0: free
    Class 1: occupied
    """

    return 2


# *********************** Processing functions
class SimpleCNN(nn.Module):
    """
    Simple CNN baseline model for parking occupancy classification.

    Input:
        RGB parking slot image with size 150x150

    Output:
        2 classes: free / occupied
    """

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Reduce feature map to 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# *********************** Main function
def main():
    number_of_classes = getNumberOfClasses()

    model = SimpleCNN(num_classes=number_of_classes)

    print("SimpleCNN model is ready.")
    print(model)

    # Test model with a dummy input image
    dummy_input = torch.randn(1, 3, 150, 150)
    dummy_output = model(dummy_input)

    print("\nDummy input shape :", dummy_input.shape)
    print("Dummy output shape:", dummy_output.shape)


if __name__ == "__main__":
    main()