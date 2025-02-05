import torch
import torch.nn as nn

class MLPTabularPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, classification=True, name='MLPTabularPredictor'):
        super(MLPTabularPredictor, self).__init__()
        self.ann = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
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
    def __init__(self, input_dim, output_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0,
                 classification=True, name='LSTMTabularPredictor'):
        super(LSTMTabularPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token
        self.num_lstm_layers = 2
        self.classification = classification
        self.name = name

        # First fully connected layer (projecting the concatenated vector of image and geotemp features)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder - will store the previous predictions in the hidden state
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                           num_layers=self.num_lstm_layers, dropout=0.2)

        # Output Layer - predicts one vector as long as num_classes for categorical tab. feature or
        # a number for numerical tab. feature
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden_state = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        cell_state   = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        tab_feature_preds = [] # predictions for the tab. feature for every horizon in the image
        #num_horizons = x[-self.max_seq_len:] # Note: should you get the number of horizons from the concatenated depths
                                              #       and only iterate through those? But then you need to also pad the predicted tab. features

        x = self.fc(x).unsqueeze(1) # (batch_size, 1, hidden_dim)
        for _ in range(self.max_seq_len):
            output, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))
            tab_feature = self.predictor(output)#.squeeze()
            if self.classification:  # only apply softmax for predicting categorical features
                tab_feature = nn.Softmax(dim=1)(tab_feature)
            tab_feature_preds.append(tab_feature)

        tab_feature_preds = torch.stack(tab_feature_preds, dim=1) # (batch_size, max_seq_len, output_dim)

        return tab_feature_preds