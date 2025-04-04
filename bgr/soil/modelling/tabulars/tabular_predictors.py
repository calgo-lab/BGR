import torch
import torch.nn as nn

class MLPTabularPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, classification=True, name='MLPTabularPredictor'):
        super(MLPTabularPredictor, self).__init__()
        self.ann = nn.Sequential(
            #nn.Linear(input_dim, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
            #nn.Linear(512, 128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim, output_dim)
        )
        self.classification = classification
        self.name = name

    def forward(self, x):
        x = self.ann(x)
        # Note: leave out Softmax for classification, since nn.CrossEntropy needs the raw logits
        #if self.classification: # only apply softmax for predicting categorical features
        #    x = nn.Softmax(dim=1)(x)
        return x


class LSTMTabularPredictor(nn.Module):
    def __init__(self,
        input_dim : int,
        output_dim : int,
        hidden_dim : int = 1024,
        num_lstm_layers : int = 2
    ):
        super(LSTMTabularPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_lstm_layers = num_lstm_layers
        
        # First fully connected layer (projecting the concatenated vector of image and geotemp features)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                            num_layers=num_lstm_layers, dropout=0.2, bidirectional=True)
        
        # Output layer
        self.predictor = nn.Linear(2*hidden_dim, output_dim)
        
    def forward(self, x : torch.Tensor):
        # x shape: (batch_size, num_segments, input_dim)
        batch_size, num_segments, _ = x.shape
        
        # Reshape for batch normalization (requires 2D input)
        x = x.view(batch_size * num_segments, -1)  # Shape: (batch_size * num_segments, input_dim)
        x = self.fc(x)  # Apply fc layer
        x = x.view(batch_size, num_segments, -1)  # Reshape back to (batch_size, num_segments, hidden_dim)
        
        # LSTM layer
        x, _ = self.lstm(x)  # Shape: (batch_size, num_segments, 2*hidden_dim)
        
        return self.predictor(x)