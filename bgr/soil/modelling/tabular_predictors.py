import torch.nn as nn

class MLPTabularPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, classification=True, name='TabularPropertyPredictor'):
        super(MLPTabularPredictor, self).__init__()
        self.ann = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.classification = classification
        self.name = name

    def forward(self, x):
        x = self.ann(x)
        if self.classification: # only apply softmax for predicting categorical features
            x = nn.Softmax(dim=1)(x)

        return x