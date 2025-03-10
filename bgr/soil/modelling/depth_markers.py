import torch
import torch.nn as nn

class LSTMDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0):
        super(LSTMDepthMarkerPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token
        self.num_lstm_layers = 2

        # First fully connected layer (projecting the concatenated vector of image and geotemp features)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder - will store the previous predictions in the hidden state
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                           num_layers=self.num_lstm_layers, dropout=0.2, bidirectional=True)

        # Output Layer - predicts one depth at a time
        self.predictor = nn.Linear(2*hidden_dim, 1)

    def forward(self, x):
        hidden_state = torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        cell_state   = torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        depth_markers = []

        x = self.fc(x).unsqueeze(1) # (batch_size, 1, hidden_dim)
        for _ in range(self.max_seq_len):
            output, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))
            depth_marker = self.predictor(output).squeeze()
            depth_markers.append(depth_marker)

        depth_markers = torch.stack(depth_markers, dim=1) # (batch_size, max_seq_len)

        # Round values very near to stop_token and above it to stop_token
        depth_markers = torch.where(
            depth_markers > self.stop_token - 0.01,
            torch.full_like(depth_markers, self.stop_token),
            depth_markers)

        return depth_markers


class TransformerDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, transformer_dim=128, num_heads=4, num_layers=2, max_seq_len=10, stop_token=1.0):
        super(TransformerDepthMarkerPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )
        self.predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(transformer_dim, max_seq_len)
        )
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token

    def forward(self, x):
        x = self.fc(x).unsqueeze(0)
        x = self.transformer(x)
        x = self.predictor(x).squeeze(-1)
        x = torch.transpose(x, 0, 1)
        x = torch.sigmoid(x).squeeze(1) # (batch_size, max_seq_len)

        # Round values very near to stop_token and above it to stop_token
        x = torch.where(
            x > self.stop_token - 0.01,
            torch.full_like(x, self.stop_token),
            x)

        return x


# Problem: this one only predicts the same number throughout the depths list.
# Aggressive regularization and adjusted loss only help insignificantly in varying the output.
# By removing the .repeat() in forward and making self.fc_out predict max_seq_len depths, there is more variance in the predicted depths list,
# but they still stay identical for all the samples. I did the same with the TransformerDepthPredictor above.
class CrossAttentionDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, transformer_dim=128, num_heads=4, num_layers=2,
                 max_seq_len=10, stop_token=1.0):
        super(CrossAttentionDepthMarkerPredictor, self).__init__()

        self.stop_token = stop_token
        self.max_seq_len = max_seq_len

        # Encoder: Processes input features (image + geotemporal features)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU()
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )

        # Decoder: Predicts depth markers step-by-step using cross-attention
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )

        # Output Layers
        self.depth_embed = nn.Embedding(max_seq_len + 1, transformer_dim)  # Embedding for positional depth tokens
        self.fc_out = nn.Linear(transformer_dim, 1)  # Predict a single depth value at each step

        self.sigmoid = nn.Sigmoid()  # Normalize output depths to [0, 1]

    def forward(self, x, targets=None):
        """
        Args:
            x: Input features (batch_size, input_dim)
            targets: Optional ground truth depth markers for teacher forcing (batch_size, seq_len)

        Returns:
            Predicted depth markers (batch_size, max_seq_len)
        """
        batch_size = x.size(0)

        # Encode input features
        x = self.encoder_fc(x)  # (batch_size, transformer_dim)
        x = x.unsqueeze(1).repeat(1, self.max_seq_len, 1).permute(1, 0, 2)  # (seq_len, batch_size, transformer_dim)
        memory = self.encoder(x)  # (seq_len, batch_size, transformer_dim)

        # Initialize decoder inputs
        sos_token = torch.zeros((batch_size, 1, 1), device=x.device)  # Start-of-sequence token
        decoder_input = self.depth_embed(sos_token.long()).squeeze(1).permute(1, 0, 2)  # (1, batch_size, transformer_dim)

        outputs = []
        for step in range(self.max_seq_len):
            # Cross-attention decoding
            decoder_output = self.decoder(decoder_input, memory)  # (step + 1, batch_size, transformer_dim)
            depth_pred = self.fc_out(decoder_output[-1])  # (batch_size, 1)
            depth_pred = self.sigmoid(depth_pred)  # Normalize to [0, 1]
            outputs.append(depth_pred)

            # Teacher forcing: use ground truth depth markers as the next input during training
            if targets is not None and step < targets.size(1):
                next_input = self.depth_embed(targets[:, step].unsqueeze(-1).long())  # (batch_size, transformer_dim)
            else:
                next_input = self.depth_embed(depth_pred.long())  # Use predicted depth marker
            decoder_input = torch.cat([decoder_input, next_input.permute(1, 0, 2)], dim=0)

        outputs = torch.cat(outputs, dim=-1)  # (batch_size, max_seq_len)
        return outputs