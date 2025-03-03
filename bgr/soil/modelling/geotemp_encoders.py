import torch.nn as nn

# Geospatial and Temporal Encoder for features in df_loc
class GeoTemporalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeoTemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)