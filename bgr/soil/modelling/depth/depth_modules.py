import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class LSTMDepthMarkerPredictorWithGuardrails(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0):
        super(LSTMDepthMarkerPredictorWithGuardrails, self).__init__()
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
        for i in range(self.max_seq_len):
            output, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))
            delta_pred = self.predictor(output).squeeze()
            
            # Guardrail: Depth marker should be positive and non-decreasing
            delta_pred = F.relu(delta_pred)
            if i > 0:
                depth_markers.append(depth_markers[-1] + delta_pred)
            else:
                # First depth marker is just the delta prediction          
                depth_markers.append(delta_pred)

        depth_markers = torch.stack(depth_markers, dim=1) # (batch_size, max_seq_len)

        # Guardrail: Depth markers should be between 0 and stop_token
        depth_markers = torch.clamp(depth_markers, min=0, max=self.stop_token)
        
        # Round values very near to stop_token and above it to stop_token
        depth_markers = torch.where(
            depth_markers > self.stop_token - 0.01,
            torch.full_like(depth_markers, self.stop_token),
            depth_markers)

        return depth_markers

class CrossAttentionTransformerDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0, num_heads=8, num_layers=2):
        super(CrossAttentionTransformerDepthMarkerPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token

        # Project input (image-geotemp vector) to hidden dimension
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Positional encoding for queries
        self.query_pos = nn.Parameter(torch.randn(max_seq_len, hidden_dim))

        # Query embeddings (learnable)
        self.query_embed = nn.Parameter(torch.randn(max_seq_len, hidden_dim))

        # Transformer decoder layers (cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Prediction head
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            depth_markers: (batch_size, max_seq_len)
        """
        batch_size = x.size(0)
        memory = self.fc(x).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Repeat the memory to match the query length
        memory = memory.repeat(1, self.max_seq_len, 1)  # (batch_size, max_seq_len, hidden_dim)

        # Prepare the query (same for all batch)
        query = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, max_seq_len, hidden_dim)

        # Apply transformer decoder
        decoded = self.transformer_decoder(query, memory)  # (batch_size, max_seq_len, hidden_dim)

        # Predict depths
        depth_markers = self.predictor(decoded).squeeze(-1)  # (batch_size, max_seq_len)

        # Round values very near to stop_token and above it to stop_token
        depth_markers = torch.where(
            depth_markers > self.stop_token - 0.01,
            torch.full_like(depth_markers, self.stop_token),
            depth_markers)

        return depth_markers

class CrossAttentionTransformerDepthMarkerPredictorWithGuardrails(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0, num_heads=8, num_layers=2):
        super(CrossAttentionTransformerDepthMarkerPredictorWithGuardrails, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token

        # Project input (image-geotemp vector) to hidden dimension
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Positional encoding for queries
        self.query_pos = nn.Parameter(torch.randn(max_seq_len, hidden_dim))

        # Query embeddings (learnable)
        self.query_embed = nn.Parameter(torch.randn(max_seq_len, hidden_dim))

        # Transformer decoder layers (cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Prediction head
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            depth_markers: (batch_size, max_seq_len)
        """
        batch_size = x.size(0)
        memory = self.fc(x).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Repeat the memory to match the query length
        memory = memory.repeat(1, self.max_seq_len, 1)  # (batch_size, max_seq_len, hidden_dim)

        # Prepare the query (same for all batch)
        query = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, max_seq_len, hidden_dim)

        # Apply transformer decoder
        decoded = self.transformer_decoder(query, memory)  # (batch_size, max_seq_len, hidden_dim)

        # Predict deltas
        pred_deltas = self.predictor(decoded).squeeze(-1)  # (batch_size, max_seq_len)
        
        # Guardrail: Ensure deltas are non-negative
        pred_deltas = F.relu(pred_deltas)
        
        # Initialize depth markers
        depth_markers = torch.zeros_like(pred_deltas)
        depth_markers[:, 0] = pred_deltas[:, 0]  # First depth marker is just the delta prediction
        
        # Guardrail: Ensure depth markers are non-decreasing
        # Compute depth markers as cumulative sum of deltas
        for i in range(1, self.max_seq_len):
            # Ensure depth markers are non-decreasing
            depth_markers[:, i] = depth_markers[:, i - 1] + pred_deltas[:, i]
        
        # Guardrail: Depth markers should be between 0 and stop_token
        depth_markers = torch.clamp(depth_markers, min=0, max=self.stop_token)
        
        # Round values very near to stop_token and above it to stop_token
        depth_markers = torch.where(
            depth_markers > self.stop_token - 0.01,
            torch.full_like(depth_markers, self.stop_token),
            depth_markers)

        return depth_markers

# DEPRECATED
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