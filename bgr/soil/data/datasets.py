import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Custom Dataset class for images, tabular data and labels
class ImageTabularDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_size=(2048, 1024),#(224, 224),  # Default size for ViT
        normalize=None,
        augment=[],
        img_path_column=None,
        label_column=None,
        geotemp_columns=None
    ):
        """
        dataframe: Pandas dataFrame with image path, table data and labels
        normalize: Operations to normalize images
        """
        self.dataframe = dataframe
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.img_path_column = img_path_column
        self.label_column = label_column
        self.geotemp_columns = geotemp_columns

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
        image_path = self.dataframe.iloc[original_idx][self.img_path_column]
        image = Image.open(image_path)
        image = transforms.Resize(self.image_size)(image)

        # Apply augmentation if specified
        if self.augment:
            image = augmentation(image)

        # Apply normalization if provided
        if self.normalize:
            image = self.normalize(image)

        # Extract geotemp features from the DataFrame (as numerical values)
        geotemp_features_array = self.dataframe.iloc[original_idx][self.geotemp_columns].astype(float).values
        geotemp_features = torch.tensor(geotemp_features_array, dtype=torch.float32)

        # Extract the label if provided
        if self.label_column:
            label = torch.tensor(self.dataframe.iloc[original_idx][self.label_column], dtype=torch.long)  # for classification (long)
            return image, geotemp_features, label
        else:
            return image, geotemp_features
        

class ImagePatchesTabularDataset(Dataset):
    """Patches from whole images.

    Args:
        
    """
    def __init__(
        self,
        dataframe,
        image_patch_size=(224, 224),
        image_patch_number=48,
        normalize=None,
        augment=[], # TODO: needed for random patches?
        img_path_column=None,
        label_column=None,
        geotemp_columns=None,
        random_state : int = None
    ):

        self.dataframe = dataframe
        self.image_patch_size = image_patch_size
        self.image_patch_number = image_patch_number
        self.normalize = normalize
        self.augment = augment
        self.img_path_column = img_path_column
        self.label_column = label_column
        self.geotemp_columns = geotemp_columns
        self.random_state = random_state

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
        image_path = self.dataframe.iloc[original_idx][self.img_path_column]
        image = Image.open(image_path)
        #image = transforms.Resize(self.image_size)(image) # no resizing, random original patches instead

        # Apply augmentation if specified
        if self.augment:
            image = augmentation(image)

        # Apply normalization if provided -> _process_image_to_patches takes care of this
        #if self.normalize:
        #    image = self.normalize(image)
        
        # Crop random patches from original image
        patches = self._process_image_to_patches(image)

        # Extract geotemp features from the DataFrame (as numerical values)
        geotemp_features_array = self.dataframe.iloc[original_idx][self.geotemp_columns].astype(float).values
        geotemp_features = torch.tensor(geotemp_features_array, dtype=torch.float32)

        # Extract the label if provided
        if self.label_column:
            label = torch.tensor(self.dataframe.iloc[original_idx][self.label_column], dtype=torch.long)  # for classification (long)
            return patches, geotemp_features, label
        else:
            return patches, geotemp_features
        
    def _process_image_to_patches(self, img : Image):
        """
        Processes an image into patches.

        Args:
            img (torch.Tensor): Image to process.

        Returns:
            torch.Tensor: Processed patches.
        """
        
        original_rng_state = torch.get_rng_state()
        if self.random_state:
            torch.manual_seed(self.random_state)
        
        # Extract patches
        random_crop = transforms.RandomCrop(self.image_patch_size, pad_if_needed=True, padding_mode='reflect')
        patches = []
        for _ in range(self.image_patch_number):
            # Randomly crop a patch
            patch = random_crop(img)
            patch = self.normalize(patch)
            patches.append(patch)
            
        # Restore the original RNG state
        torch.set_rng_state(original_rng_state)
        
        # Convert patches to tensor
        patches = torch.stack(patches)
        
        return patches
        

class SegmentsTabularDataset(Dataset):
    """
    TODO: Add docstring
    """
    def __init__(
        self,
        dataframe,
        segment_size=(512, 1024),
        normalize=None,
        img_path_column : str ='file',
        depth_column : str ='Untergrenze',
        label_column : str ='Horizontsymbol_relevant', # TODO: Maybe this doesnt work?
        max_segments : int =8,
        geotemp_columns : list =None,
        tab_num_columns : list = None,
        tab_categ_columns : dict =None,
        image_size : tuple = (2048, 1024)
    ):
        """
        
        """
        self.dataframe = dataframe
        self.segment_size = segment_size
        self.normalize = normalize
        self.img_path_column = img_path_column
        self.depth_column = depth_column
        self.label_column = label_column
        self.max_segments = max_segments
        self.geotemp_columns = geotemp_columns
        self.tab_num_columns = tab_num_columns
        self.tab_categ_columns = tab_categ_columns
        self.image_size = image_size
        
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
        image_path = row[self.img_path_column]
        image = Image.open(image_path)
        
        # Convert normalized depth markers to pixel indices
        pixel_depths = [int(depth * image.height) for depth in [0.0] + row[self.depth_column]]  # Add 0.0 for upmost bound
            
        # Crop to segments
        segments = []
        labels = []
        tabular_features = []
        for i in range(len(pixel_depths) - 1):
            upper, lower = pixel_depths[i], pixel_depths[i + 1]
            
            # Crop and resize the segment
            segment = image.crop((0, upper, image.width, lower))
            segment = segment.resize(self.segment_size)
            segment = self.normalize(segment)
            segments.append(segment)
            
            # Extract segment-specific tabular features
            if self.tab_num_columns:
                num_tabular_features_array = [row[feature][i] for feature in self.tab_num_columns]
                num_tabular_features = torch.tensor(num_tabular_features_array, dtype=torch.float32)
                tabular_features.append(num_tabular_features)
                
            # One hot encode categorical features
            if self.tab_categ_columns:
                categ_tabular_features_array = [row[feature][i] for feature in self.tab_categ_columns.keys()]
                
                # [2, 3, 5] -> [0, 0, 1, ... , 0, 0, 0, 1, ... , 0, 0, 0, 0, 0, 1, ...]
                onehot_encoded_tabular_feature_array = np.zeros(sum(self.tab_categ_columns.values()))
                cum_sum = 0
                for idx, value in enumerate(categ_tabular_features_array):
                    onehot_encoded_tabular_feature_array[cum_sum + value] = 1
                    cum_sum += list(self.tab_categ_columns.values())[idx]
                
                onehot_tabular_features = torch.tensor(onehot_encoded_tabular_feature_array, dtype=torch.long)
                tabular_features[i] = torch.cat([tabular_features[i], onehot_tabular_features], dim=0)
            
            # Extract the depth and label
            label = torch.tensor(row[self.label_column][i], dtype=torch.long)
            labels.append(label)
        
        # Pad segments, segments tabular and labels to ensure consistent sizes
        while len(segments) < self.max_segments:
            # Pad segments images with zeros
            segments.append(torch.zeros_like(segments[0]))
            
            # Pad segments tabular features with zeros
            if self.tab_num_columns or self.tab_categ_columns:
                tabular_features.append(torch.zeros_like(tabular_features[0]))
            
            # Pad labels with -1
            labels.append(torch.tensor(-1, dtype=torch.long))  # Use -1 as a padding label

        # Convert segments, segments tabular features and labels to tensors
        segments = torch.stack(segments)
        if self.tab_num_columns or self.tab_categ_columns:
            tabular_features = torch.stack(tabular_features)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Resize and normalize the whole image (only needed when training end-to-end)
        image = transforms.Resize(self.image_size)(image)
        image = self.normalize(image)

        if self.geotemp_columns:
            # Extract geotemp features from the DataFrame (as numerical values)
            geotemp_features_array = row[self.geotemp_columns].astype(float).values
            geotemp_features = torch.tensor(geotemp_features_array, dtype=torch.float32)
        
            if self.tab_num_columns or self.tab_categ_columns:
                # Non-empty image, segments, tabular_features, geotemp_features, labels
                pass
            else:
                # Non-empty image segments, geotemp_features, labels
                tabular_features = []
        else:
            if self.tab_num_columns:
                # Non-empty image segments, tabular_features, labels
                geotemp_features = []
            else:
                # Non-empty image segments, labels
                tabular_features, geotemp_features = [], []
            
        # Always return everything, even if some are empty
        return image, segments, tabular_features, geotemp_features, labels
            
class SegmentPatchesTabularDataset(Dataset):
    """
    Custom Dataset class for handling image segments into patches by RandomCrop and tabular data.

    
    """
    def __init__(
        self,
        dataframe,
        segment_patch_size=(224, 224),
        segment_patch_number=48,
        normalize=None,
        img_path_column : str ='file',
        depth_column : str ='Untergrenze',
        label_column : str ='Horizontsymbol_relevant', # TODO: Maybe this doesnt work?
        max_segments : int =8,
        geotemp_columns : list =None,
        tab_num_columns : list = None,
        tab_categ_columns : dict =None,
        random_state : int = None,
        image_size : tuple = (2048, 1024)
    ):
        """
        Initializes the SegmentsTabularDataset.

        
        """
        self.dataframe = dataframe
        self.segment_patch_size = segment_patch_size
        self.segment_patch_number = segment_patch_number
        self.normalize = normalize
        self.img_path_column = img_path_column
        self.depth_column = depth_column
        self.label_column = label_column
        self.max_segments = max_segments
        self.geotemp_columns = geotemp_columns
        self.tab_num_columns = tab_num_columns
        self.tab_categ_columns = tab_categ_columns
        self.random_state = random_state
        self.image_size = image_size
        
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
        image_path = row[self.img_path_column]
        image = Image.open(image_path)
        
        # Convert normalized depth markers to pixel indices
        pixel_depths = [int(depth * image.height) for depth in [0.0] + row[self.depth_column]]  # Add 0.0 for upmost bound
            
        # Crop to segments
        segment_patches = []
        labels = []
        tabular_features = []
        for i in range(len(pixel_depths) - 1):
            upper, lower = pixel_depths[i], pixel_depths[i + 1]
            
            # Crop and process the segment into patches
            segment = image.crop((0, upper, image.width, lower))
            patches = self._process_segment_to_patches(segment)
            segment_patches.append(patches)
            
            # Extract segment-specific tabular features
            if self.tab_num_columns:
                num_tabular_features_array = [row[feature][i] for feature in self.tab_num_columns]
                num_tabular_features = torch.tensor(num_tabular_features_array, dtype=torch.float32)
                tabular_features.append(num_tabular_features)
                
            # One hot encode categorical features
            if self.tab_categ_columns:
                categ_tabular_features_array = [row[feature][i] for feature in self.tab_categ_columns.keys()]
                
                # [2, 3, 5] -> [0, 0, 1, ... , 0, 0, 0, 1, ... , 0, 0, 0, 0, 0, 1, ...]
                onehot_encoded_tabular_feature_array = np.zeros(sum(self.tab_categ_columns.values()))
                cum_sum = 0
                for idx, value in enumerate(categ_tabular_features_array):
                    onehot_encoded_tabular_feature_array[cum_sum + value] = 1
                    cum_sum += list(self.tab_categ_columns.values())[idx]
                
                onehot_tabular_features = torch.tensor(onehot_encoded_tabular_feature_array, dtype=torch.long)
                tabular_features[i] = torch.cat([tabular_features[i], onehot_tabular_features], dim=0)
            
            # Extract the depth and label
            label = torch.tensor(row[self.label_column][i], dtype=torch.long)
            labels.append(label)
        
        # Pad segments, segments tabular and labels to ensure consistent sizes
        while len(segment_patches) < self.max_segments:
            # Pad segments patch images with zeros
            segment_patches.append(torch.zeros_like(segment_patches[0]))
            
            # Pad segments tabular features with zeros
            if self.tab_num_columns or self.tab_categ_columns:
                tabular_features.append(torch.zeros_like(tabular_features[0]))
            
            # Pad labels with -1
            labels.append(torch.tensor(-1, dtype=torch.long))  # Use -1 as a padding label

        # Convert segments, segments tabular features and labels to tensors
        segment_patches = torch.stack(segment_patches)
        if self.tab_num_columns or self.tab_categ_columns:
            tabular_features = torch.stack(tabular_features)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Resize and normalize the whole image (only needed when training end-to-end)
        image = transforms.Resize(self.image_size)(image)
        image = self.normalize(image)

        if self.geotemp_columns:
            # Extract geotemp features from the DataFrame (as numerical values)
            geotemp_features_array = row[self.geotemp_columns].astype(float).values
            geotemp_features = torch.tensor(geotemp_features_array, dtype=torch.float32)
        
            if self.tab_num_columns or self.tab_categ_columns:
                # Non-empty image segments, tabular_features, geotemp_features, labels
                pass
            else:
                # Non-empty image segments, geotemp_features, labels
                tabular_features = []
        else:
            if self.tab_num_columns:
                # Non-empty image segments, tabular_features, labels
                geotemp_features = []
            else:
                # Non-empty image segments, labels
                tabular_features, geotemp_features = [], []
            
        # Always return everything, even if some are empty
        return image, segment_patches, tabular_features, geotemp_features, labels
    
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
            patch = self.normalize(patch)
            patches.append(patch)
            
        # Restore the original RNG state
        torch.set_rng_state(original_rng_state)
        
        # Convert patches to tensor
        patches = torch.stack(patches)
        
        return patches