import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def simplify_string(complex_string, mapping_df, col_name):
    """Simplifies a complex string based on the mapping

    :param complex_string:
    :param mapping_df:
    :return:
    """
    complex_symb = complex_string.split('; ')[1]
    # Find the matching simplified part in the mapping DataFrame
    simple_symb_series = mapping_df[col_name][mapping_df['Horiz'] == complex_symb]

    # Keep the complex symbol, when there is no simplified alternative
    if not simple_symb_series.empty:
        simple_symb = simple_symb_series.values[0]
    else:
        simple_symb = complex_symb

    return simple_symb


def encode_categorical_columns(df, col_name):
    """

    :param df:
    :param col_name:
    :return:
    """
    counts = df[col_name].value_counts()
    df[col_name] = df[col_name].replace(counts.index, range(len(counts)))


def normalize_df_list(lst, max_boundary=100.0):
    """
    Rounds the last list item to a max. threshold and normalizes the whole list.
    :param lst:
    :param max_boundary:
    :return:
    """
    lst[-1] = max_boundary
    lst = [x/max_boundary for x in lst]
    return lst


# Custom Dataset class for images, tabular data and labels
class ImageTabularDataset(Dataset):
    def __init__(self,
                 dataframe,
                 image_size=(224, 224),  # Default size for ViT
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

