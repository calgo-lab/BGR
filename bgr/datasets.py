import torch
from torch.utils.data import Dataset
from PIL import Image


def simplify_string(complex_string, mapping_df, col_name = 'stark vereinfacht'):
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


# Custom Dataset class für Bilder, tabellarische Daten und Labels
class ImageTabularDataset(Dataset):
    def __init__(self,
                 dataframe,
                 transform=None,
                 image_path=None,
                 label=None,
                 feature_columns=None
                 ):
        """
        dataframe: Pandas DataFrame mit Bildpfaden, tabellarischen Daten und Labels
        transform: Bildtransformationen
        """
        self.dataframe = dataframe
        self.transform = transform
        self.image_path = image_path
        self.label = label
        self.feature_columns = feature_columns

    def __len__(self):
        # Länge des Datasets (Anzahl der Zeilen im DataFrame)
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extrahiere den Bildpfad aus dem DataFrame
        image_path = self.dataframe.iloc[idx][self.image_path]

        # Lade das Bild
        image = Image.open(image_path)

        # Wende Bildtransformationen an, falls vorhanden
        if self.transform:
            image = self.transform(image)

        # Extrahiere die tabellarischen Daten (numerische Features) aus dem DataFrame
        tabular_features_array = self.dataframe.iloc[idx][self.feature_columns].astype(float).values
        tabular_features = torch.tensor(tabular_features_array, dtype=torch.float32)

        # Extrahiere das Label
        label = torch.tensor(self.dataframe.iloc[idx][self.label], dtype=torch.long)  # Für Klassifikation (long)

        return image, tabular_features, label