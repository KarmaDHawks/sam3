import torch.nn as nn

#"""
class FeatureCorrector(nn.Module):
    def __init__(self, C=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


def get_mlp_corrector():
    return FeatureCorrector()