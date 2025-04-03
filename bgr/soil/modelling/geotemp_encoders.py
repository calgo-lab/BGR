import torch.nn as nn

# Geospatial and Temporal Encoder for features in df_loc
class GeoTemporalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        
        super(GeoTemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)