import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Custom Dataset class for images, tabular data and labels
class ImageTabularDataset(Dataset):
    def __init__(self,
                 dataframe,
                 image_size=(2048, 1024),#(224, 224),  # Default size for ViT
                 normalize=None,
                 augment=[],
                 image_path=None,
                 label=None,
                 feature_columns=None
                 ):
        """
        dataframe: Pandas dataFrame with image path, table data and labels
        normalize: Operations to normalize images
        """
        self.dataframe = dataframe
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.image_path = image_path
        self.label = label
        self.feature_columns = feature_columns

        # Precompute a list of (index, augmentation) tuples to represent the expanded dataset
        self.index_map = []
        for idx in range(len(self.dataframe)):
            # Original image
            self.index_map.append((idx, lambda x: x)) # Identity function for no augmentation
            # Augmented images
            for aug in self.augment:
                self.index_map.append((idx, aug))

    def __len__(self):
        # Dataset length (number of rows with or without augmentation)
        return len(self.index_map)

    def __getitem__(self, expanded_idx):
        # Get the original dataframe index and the augmentation to apply
        original_idx, augmentation = self.index_map[expanded_idx]

        # Extract the image path from the DataFrame, read and resize image
        image_path = self.dataframe.iloc[original_idx][self.image_path]
        image = Image.open(image_path)
        image = transforms.Resize(self.image_size)(image)

        # Apply augmentation if specified
        if self.augment:
            image = augmentation(image)

        # Apply normalization if provided
        if self.normalize:
            image = self.normalize(image)

        # Extract tabular features from the DataFrame (as numerical values)
        tabular_features_array = self.dataframe.iloc[original_idx][self.feature_columns].astype(float).values
        tabular_features = torch.tensor(tabular_features_array, dtype=torch.float32)

        # Extract the label if provided
        if self.label:
            label = torch.tensor(self.dataframe.iloc[original_idx][self.label], dtype=torch.long)  # for classification (long)
            return image, tabular_features, label
        else:
            return image, tabular_features

