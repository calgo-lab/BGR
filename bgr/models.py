import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


# Definiere das Modell mit DINOv2 Backbone von Hugging Face
class ImageTabularModel(nn.Module):
    def __init__(self, vision_backbone, num_tabular_features, num_classes):
        super(ImageTabularModel, self).__init__()

        # Lade das vortrainierte DINOv2-Modell von Hugging Face
        self.vision_backbone = AutoModel.from_pretrained(vision_backbone)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_backbone)

        # Definiere ein MLP für die tabellarischen Daten
        self.fc_tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Kombinierte Fully Connected Layers
        self.fc_combined = nn.Sequential(
            # nn.Linear(768 + 16, 64),  # 768 ist die Ausgabegröße von DINOv2, 16 ist die Größe des MLP
            nn.Linear(384 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, tabular_features):
        # Extrahiere die Bildfeatures
        image_features = self.vision_backbone(pixel_values=image).last_hidden_state[:, 0, :]
        # import pdb;pdb.set_trace()

        # Verarbeite die tabellarischen Daten mit dem MLP
        tabular_features_processed = self.fc_tabular(tabular_features)

        # Kombiniere Bild- und Tabellarische Features
        combined_features = torch.cat((image_features, tabular_features_processed), dim=1)

        # Endgültige Vorhersage
        output = self.fc_combined(combined_features)
        return output