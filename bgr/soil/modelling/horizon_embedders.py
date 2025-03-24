import torch
import torch.nn as nn

class HorizonEmbedder(nn.Module):
    """MLP predictor of horizon embeddings. Returns one embedding per segment.
    """
    def __init__(self, input_dim, output_dim):
        super(HorizonEmbedder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
    
    
class HorizonLSTMEmbedder(nn.Module):
    """LSTM predictor of horizon embeddings. Returns all embeddings for all segments.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(HorizonLSTMEmbedder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim # dimension of one embedding vector
        self.hidden_dim = hidden_dim
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

        # Output Layer - predicts one embedding at a time
        self.predictor = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x, num_segments):
        #hidden_state = torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        #cell_state   = torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        horiz_embeddings = [] # list with all the embeddings for the horizons in the sample

        # Apply self.fc to each vector in x - initially (batch_size*num_segments, input_dim)
        x = self.fc(x).view(-1, num_segments, self.hidden_dim)  # Reshape back to (batch_size, num_segments, hidden_dim)

        # Process with self.rnn in chunks of num_segments
        #for i in range(0, num_segments*batch_size, num_segments):
        #    input_chunk = x[:, i:i + num_segments, :]  # features for all segments in the current sample of the batch
        #for _ in range(num_segments):
        output, (hidden_state, cell_state) = self.rnn(x) # (batch_size, num_segments, 2*hidden_dim)
        horiz_embeddings = self.predictor(output)  # (bacth_size, num_segments, output_dim)
        #horiz_embeddings.append(horiz_emb)

        #horiz_embeddings = torch.stack(horiz_embeddings, dim=1) # (batch_size, num_segments, output_dim)

        # We are using zero vectors for padding. Round values very close to 0 to 0
        #horiz_embeddings = torch.where(
        #    -1e-12 < horiz_embeddings < 1e-12,
        #    torch.full_like(horiz_embeddings, 0.0),
        #    horiz_embeddings)

        return horiz_embeddings