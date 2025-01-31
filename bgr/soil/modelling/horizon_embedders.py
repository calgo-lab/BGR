import torch.nn as nn

class HorizonEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HorizonEmbedder, self).__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.embedder(x)