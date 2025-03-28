import numpy as np
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
    """
    Custom Dataset class for handling image segments and tabular data.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image paths, depth markers, and labels.
        segment_size (tuple): Size to which each image segment will be resized.
        normalize (callable): Transformation to normalize images.
        path_column (str): Column name for image paths in the DataFrame.
        depth_column (str): Column name for depth markers in the DataFrame.
        label_column (str): Column name for labels in the DataFrame.
        max_segments (int): Maximum number of segments per image.
        feature_columns (list): List of column names for tabular features.
        segments_tab_feature_columns (list): List of tabular features for each segment.
    """
    def __init__(
        self,
        dataframe,
        segment_size=(512, 1024),
        normalize=None,
        path_column : str ='file',
        depth_column : str ='Untergrenze',
        label_column : str ='Horizontsymbol_relevant', # TODO: Maybe this doesnt work?
        max_segments : int =8,
        feature_columns : list =None,
        segments_tab_num_feature_columns : list = None,
        segments_tab_categ_feature_columns : dict =None
    ):
        """
        Initializes the SegmentsTabularDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths, depth markers, and labels.
            segment_size (tuple): Size to which each image segment will be resized.
            normalize (callable, optional): Transformation to normalize images. Defaults to None.
            path_column (str): Column name for image paths in the DataFrame.
            depth_column (str): Column name for depth markers in the DataFrame.
            label_column (str): Column name for labels in the DataFrame.
            max_segments (int): Maximum number of segments per image.
            feature_columns (list, optional): List of column names for tabular features. Defaults to None.
            segments_tab_num_feature_columns (list, optional): List of column names for segment-specific tabular features. Defaults to None.
            segments_tab_categ_feature_columns (dict, optional): Dictionary of column names for segment-specific categorical features, with the number of categories as values. Defaults to None.
        """
        self.dataframe = dataframe
        self.segment_size = segment_size
        self.normalize = normalize
        self.path_column = path_column
        self.depth_column = depth_column
        self.label_column = label_column
        self.max_segments = max_segments
        self.feature_columns = feature_columns
        self.segments_tabular_num_features = segments_tab_num_feature_columns
        self.segments_tab_categ_features = segments_tab_categ_feature_columns
        
        if self.normalize is None:
            self.normalize = transforms.Compose([
                transforms.ToTensor()
            ])
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing segments, tabular features (if available), and labels.
        """
        row = self.dataframe.iloc[index]
        
        # Extract the image path from the DataFrame, read and resize image
        image_path = row[self.path_column]
        image = Image.open(image_path)
        
        # Convert normalized depth markers to pixel indices
        pixel_depths = [int(depth * image.height) for depth in [0.0] + row[self.depth_column]]  # Add 0.0 for upmost bound
            
        # Crop to segments
        segments = []
        labels = []
        segments_specific_tabular_features = []
        for i in range(len(pixel_depths) - 1):
            upper, lower = pixel_depths[i], pixel_depths[i + 1]
            
            # Crop and resize the segment
            segment = image.crop((0, upper, image.width, lower))
            segment = segment.resize(self.segment_size)
            segment = self.normalize(segment)
            segments.append(segment)
            
            # Extract segment-specific tabular features
            if self.segments_tabular_num_features:
                segment_tabular_features_array = [row[feature][i] for feature in self.segments_tabular_num_features]
                segment_tabular_features = torch.tensor(segment_tabular_features_array, dtype=torch.float32)
                segments_specific_tabular_features.append(segment_tabular_features)
                
            # One hot encode categorical features
            if self.segments_tab_categ_features:
                segment_tabular_features_array = [row[feature][i] for feature in self.segments_tab_categ_features.keys()]
                
                # [2, 3, 5] -> [0, 0, 1, ... , 0, 0, 0, 1, ... , 0, 0, 0, 0, 0, 1, ...]
                onehot_encoded_tabular_feature_array = np.zeros(sum(self.segments_tab_categ_features.values()))
                cum_sum = 0
                for idx, value in enumerate(segment_tabular_features_array):
                    onehot_encoded_tabular_feature_array[cum_sum + value] = 1
                    cum_sum += list(self.segments_tab_categ_features.values())[idx]
                
                segment_onehot_tabular_features = torch.tensor(onehot_encoded_tabular_feature_array, dtype=torch.long)
                segments_specific_tabular_features[i] = torch.cat([segments_specific_tabular_features[i], segment_onehot_tabular_features], dim=0)
            
            # Extract the depth and label
            label = torch.tensor(row[self.label_column][i], dtype=torch.long)
            labels.append(label)
        
        # Pad segments, segments tabular and labels to ensure consistent sizes
        while len(segments) < self.max_segments:
            # Pad segments images with zeros
            segments.append(torch.zeros_like(segments[0]))
            
            # Pad segments tabular features with zeros
            if self.segments_tabular_num_features:
                segments_specific_tabular_features.append(torch.zeros_like(segments_specific_tabular_features[0]))
            
            # Pad labels with -1
            labels.append(torch.tensor(-1, dtype=torch.long))  # Use -1 as a padding label

        # Convert segments, segments tabular features and labels to tensors
        segments = torch.stack(segments)
        if self.segments_tabular_num_features:
            segments_specific_tabular_features = torch.stack(segments_specific_tabular_features)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.feature_columns:
            # Extract tabular features from the DataFrame (as numerical values)
            tabular_features_array = row[self.feature_columns].astype(float).values
            tabular_features = torch.tensor(tabular_features_array, dtype=torch.float32)
        
            if self.segments_tabular_num_features:
                return segments, segments_specific_tabular_features, tabular_features, labels
            else:
                return segments, tabular_features, labels
        else:
            if self.segments_tabular_num_features:
                return segments, segments_specific_tabular_features, labels
            else:
                return segments, labels
            
class SegmentPatchesTabularDataset(Dataset):
    """
    Custom Dataset class for handling image segments into patches by RandomCrop and tabular data.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image paths, depth markers, and labels.
        segment_size (tuple): Size to which each image segment will be resized.
        normalize (callable): Transformation to normalize images.
        path_column (str): Column name for image paths in the DataFrame.
        depth_column (str): Column name for depth markers in the DataFrame.
        label_column (str): Column name for labels in the DataFrame.
        max_segments (int): Maximum number of segments per image.
        feature_columns (list): List of column names for tabular features.
        segments_tab_feature_columns (list): List of tabular features for each segment.
    """
    def __init__(
        self,
        dataframe,
        segment_patch_size=(224, 224),
        segment_patch_number=64,
        normalize=None,
        path_column : str ='file',
        depth_column : str ='Untergrenze',
        label_column : str ='Horizontsymbol_relevant', # TODO: Maybe this doesnt work?
        max_segments : int =8,
        feature_columns : list =None,
        segments_tab_num_feature_columns : list = None,
        segments_tab_categ_feature_columns : dict =None,
        random_state : int = None
    ):
        """
        Initializes the SegmentsTabularDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths, depth markers, and labels.
            segment_size (tuple): Size to which each image segment will be resized.
            normalize (callable, optional): Transformation to normalize images. Defaults to None.
            path_column (str): Column name for image paths in the DataFrame.
            depth_column (str): Column name for depth markers in the DataFrame.
            label_column (str): Column name for labels in the DataFrame.
            max_segments (int): Maximum number of segments per image.
            feature_columns (list, optional): List of column names for tabular features. Defaults to None.
            segments_tab_num_feature_columns (list, optional): List of column names for segment-specific tabular features. Defaults to None.
            segments_tab_categ_feature_columns (dict, optional): Dictionary of column names for segment-specific categorical features, with the number of categories as values. Defaults to None.
            random_state (int, optional): Random seed for the Random Patch Cropping reproducibility. Defaults to None.
        """
        self.dataframe = dataframe
        self.segment_patch_size = segment_patch_size
        self.segment_patch_number = segment_patch_number
        self.normalize = normalize
        self.path_column = path_column
        self.depth_column = depth_column
        self.label_column = label_column
        self.max_segments = max_segments
        self.feature_columns = feature_columns
        self.segments_tabular_num_features = segments_tab_num_feature_columns
        self.segments_tab_categ_features = segments_tab_categ_feature_columns
        self.random_state = random_state
        
        if self.normalize is None:
            self.normalize = transforms.Compose([
                transforms.ToTensor()
            ])
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing segments, tabular features (if available), and labels.
        """
        row = self.dataframe.iloc[index]
        
        # Extract the image path from the DataFrame, read and resize image
        image_path = row[self.path_column]
        image = Image.open(image_path)
        
        # Convert normalized depth markers to pixel indices
        pixel_depths = [int(depth * image.height) for depth in [0.0] + row[self.depth_column]]  # Add 0.0 for upmost bound
            
        # Crop to segments
        segment_patches = []
        labels = []
        segments_specific_tabular_features = []
        for i in range(len(pixel_depths) - 1):
            upper, lower = pixel_depths[i], pixel_depths[i + 1]
            
            # Crop and and process the segment into patches
            segment = image.crop((0, upper, image.width, lower))
            patches = self._process_segment_to_patches(segment)
            segment_patches.append(patches)
            
            # Extract segment-specific tabular features
            if self.segments_tabular_num_features:
                segment_tabular_features_array = [row[feature][i] for feature in self.segments_tabular_num_features]
                segment_tabular_features = torch.tensor(segment_tabular_features_array, dtype=torch.float32)
                segments_specific_tabular_features.append(segment_tabular_features)
                
            # One hot encode categorical features
            if self.segments_tab_categ_features:
                segment_tabular_features_array = [row[feature][i] for feature in self.segments_tab_categ_features.keys()]
                
                # [2, 3, 5] -> [0, 0, 1, ... , 0, 0, 0, 1, ... , 0, 0, 0, 0, 0, 1, ...]
                onehot_encoded_tabular_feature_array = np.zeros(sum(self.segments_tab_categ_features.values()))
                cum_sum = 0
                for idx, value in enumerate(segment_tabular_features_array):
                    onehot_encoded_tabular_feature_array[cum_sum + value] = 1
                    cum_sum += list(self.segments_tab_categ_features.values())[idx]
                
                segment_onehot_tabular_features = torch.tensor(onehot_encoded_tabular_feature_array, dtype=torch.long)
                segments_specific_tabular_features[i] = torch.cat([segments_specific_tabular_features[i], segment_onehot_tabular_features], dim=0)
            
            # Extract the depth and label
            label = torch.tensor(row[self.label_column][i], dtype=torch.long)
            labels.append(label)
        
        # Pad segments, segments tabular and labels to ensure consistent sizes
        while len(segment_patches) < self.max_segments:
            # Pad segments patch images with zeros
            segment_patches.append(torch.zeros_like(segment_patches[0]))
            
            # Pad segments tabular features with zeros
            if self.segments_tabular_num_features:
                segments_specific_tabular_features.append(torch.zeros_like(segments_specific_tabular_features[0]))
            
            # Pad labels with -1
            labels.append(torch.tensor(-1, dtype=torch.long))  # Use -1 as a padding label

        # Convert segments, segments tabular features and labels to tensors
        segment_patches = torch.stack(segment_patches)
        if self.segments_tabular_num_features:
            segments_specific_tabular_features = torch.stack(segments_specific_tabular_features)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.feature_columns:
            # Extract tabular features from the DataFrame (as numerical values)
            tabular_features_array = row[self.feature_columns].astype(float).values
            tabular_features = torch.tensor(tabular_features_array, dtype=torch.float32)
        
            if self.segments_tabular_num_features:
                return segment_patches, segments_specific_tabular_features, tabular_features, labels
            else:
                return segment_patches, tabular_features, labels
        else:
            if self.segments_tabular_num_features:
                return segment_patches, segments_specific_tabular_features, labels
            else:
                return segment_patches, labels
    
    def _process_segment_to_patches(self, segment : Image):
        """
        Processes a segment into patches.

        Args:
            segment (torch.Tensor): Segment to process.

        Returns:
            torch.Tensor: Processed patches.
        """
        
        # if segment.height < self.segment_patch_size[0]:
        #     # Resize the segment to ensure the height is at least the patch size
        #     resize_transform = transforms.Resize((self.segment_patch_size[0], int(segment.width * self.segment_patch_size[0] / segment.height)))
        #     segment = resize_transform(segment)
        
        original_rng_state = torch.get_rng_state()
        if self.random_state:
            torch.manual_seed(self.random_state)
        
        # Extract patches
        random_crop = transforms.RandomCrop(self.segment_patch_size, pad_if_needed=True, padding_mode='reflect')
        patches = []
        for _ in range(self.segment_patch_number):
            # Randomly crop a patch
            patch = random_crop(segment)
            self.normalize(patch)
            patches.append(patch)
            
        # Restore the original RNG state
        torch.set_rng_state(original_rng_state)
        
        # Convert patches to tensor
        patches = torch.stack(patches)
        
        return patches