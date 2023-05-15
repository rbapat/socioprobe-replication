import torch.nn as nn


class SimpleProbe(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(SimpleProbe, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.model(x)
