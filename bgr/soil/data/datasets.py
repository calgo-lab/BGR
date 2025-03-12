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

class SegmentsTabularDataset(Dataset):
    def __init__(
        self,
        dataframe,
        segment_size=(512, 1024),
        normalize=None,
        path_column='file',
        depth_column='Untergrenze',
        label_column='Horizontsymbol_relevant', # TODO: Maybe this doesnt work?
        max_segments=8,
        feature_columns=None
    ):
        self.dataframe = dataframe
        self.segment_size = segment_size
        self.normalize = normalize
        self.path_column = path_column
        self.depth_column = depth_column
        self.label_column = label_column
        self.max_segments = max_segments
        self.feature_columns = feature_columns
        
        if self.normalize is None:
            self.normalize = transforms.Compose([
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        # Extract the image path from the DataFrame, read and resize image
        image_path = row[self.path_column]
        image = Image.open(image_path)
        
        # Convert normalized depth markers to pixel indices
        pixel_depths = [int(depth * image.height) for depth in [0.0] + row[self.depth_column]]  # Add 0.0 for upmost bound
            
        # Crop to segments
        segments = []
        labels = []
        for i in range(len(pixel_depths) - 1):
            upper, lower = pixel_depths[i], pixel_depths[i + 1]
            
            # Crop and resize the segment
            segment = image.crop((0, upper, image.width, lower))
            segment = segment.resize(self.segment_size)
            segment = self.normalize(segment)
            segments.append(segment)
        
            # Extract the depth and label
            label = torch.tensor(row[self.label_column][i], dtype=torch.long)
            labels.append(label)
        
        # Pad segments and labels to ensure consistent sizes
        while len(segments) < self.max_segments:
            segments.append(torch.zeros_like(segments[0]))
            labels.append(torch.tensor(-1, dtype=torch.long))  # Use -1 as a padding label

        # Convert segments and labels to tensors
        segments = torch.stack(segments)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.feature_columns:
            # Extract tabular features from the DataFrame (as numerical values)
            tabular_features_array = row[self.feature_columns].astype(float).values
            tabular_features = torch.tensor(tabular_features_array, dtype=torch.float32)
        
            return segments, tabular_features, labels
        else:
            return segments, labels