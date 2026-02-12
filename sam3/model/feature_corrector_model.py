import torch.nn as nn

#"""
class FeatureCorrector(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self.cnn(x)


def get_mlp_corrector():
    return FeatureCorrector()