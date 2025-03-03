import torch.nn as nn

class HorizonEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HorizonEmbedder, self).__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.embedder(x)