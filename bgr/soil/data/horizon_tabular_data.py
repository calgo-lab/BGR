from typing import Tuple
from jellyfish import levenshtein_distance
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

FILE_NAME = "data_horizons.csv"
SIMPLIFIED_SYMBOL_FILE_NAME = "Vereinfachung_Horizontsymbole.csv"
IMAGE_FOLDER_NAME = "Profilbilder_no_ruler_no_sky/"
LOCATIONS_FILE_NAME = "STANDORT.csv"

class HorizonDataProcessor:
    """
    Processes horizon data for soil analysis, including loading, preprocessing, merging, 
    filtering, imputing, encoding, and scaling features.
    
    Args:
        label_embeddings_path (str): Path to the label embeddings file.
        data_folder_path (str): Path to the data folder. Default is "../data/BGR/".
    """
    def __init__(self, label_embeddings_path: str, data_folder_path: str = "../data/BGR/"):
        # Validate provided paths and required files/folders
        self._validate_paths(label_embeddings_path, data_folder_path)
        
        self.data_folder_path = data_folder_path
        self.label_embeddings_path = label_embeddings_path
        
        # Define target and feature columns as class attributes for external access
        self.target = 'Horizontsymbol_relevant'
        self.num_features = [
            'xcoord', 'ycoord', 'Steine', 'GrundwaStand', 
            'Moormaechtigkeit', 'Torfmaechtigkeit', 'KV_0_30', 'KV_30_100'
        ]
        self.categ_features = [
            'Probenahme_Monat', 'Probenahme_Jahr', 'Bodenart', 'Bodenfarbe', 'Karbonat', 
            'Humusgehaltsklasse', 'Durchwurzelung', 'Bodenklimaraum_Name', 'Landnutzung', 
            'BZE_Moor', 'Hauptbodentyp', 'GrundwaStufe', 'Neigung', 'Exposition', 'Woelbung', 
            'Reliefformtyp', 'LageImRelief'
        ]
        self.keep_columns = [
            'Point', 'Obergrenze', 'Untergrenze', self.target, 
            'Bodenart', 'Bodenfarbe', 'Steine', 'Karbonat', 'Humusgehaltsklasse', 'Durchwurzelung', 
            'file', 'Probenahme_Monat', 'Probenahme_Jahr', 'xcoord', 'ycoord', 'Bodenklimaraum_Name',
            'Landnutzung', 'BZE_Moor', 'Hauptbodentyp', 'GrundwaStufe', 'GrundwaStand', 
            'Moormaechtigkeit', 'Torfmaechtigkeit', 'Neigung', 'Exposition', 'Woelbung', 
            'Reliefformtyp', 'LageImRelief', 'KV_0_30', 'KV_30_100'
        ]
        self.soil_infos = ['Bodenart', 'Bodenfarbe', 'Steine', 'Karbonat', 'Humusgehaltsklasse', 'Durchwurzelung']
        self.geotemp_img_infos = [
            'Probenahme_Monat', 'Probenahme_Jahr', 'xcoord', 'ycoord', 'Bodenklimaraum_Name',
            'Landnutzung', 'BZE_Moor', 'Hauptbodentyp', 'GrundwaStufe', 'GrundwaStand',
            'Moormaechtigkeit', 'Torfmaechtigkeit', 'Neigung', 'Exposition', 'Woelbung', 
            'Reliefformtyp', 'LageImRelief', 'KV_0_30', 'KV_30_100', 'file'
        ]
        # For now, leave out Bodenart and Bodenfarbe. Also, we don't need to stratify wrt stones (it's numerical)
        self.stratified_split_targets = ['Karbonat', 'Humusgehaltsklasse', 'Durchwurzelung', 'Horizontsymbol_relevant']
        
    @staticmethod
    def _validate_paths(label_embeddings_path: str, data_folder_path: str) -> None:
        """
        Validates the provided paths and required files/folders.
        
        Args:
            label_embeddings_path (str): Path to the label embeddings file.
            data_folder_path (str): Path to the data folder.
        
        Raises:
            FileNotFoundError: If the label embeddings file or required data files are not found.
            ValueError: If the label embeddings file does not have a .pkl or .pickle extension.
            NotADirectoryError: If the data folder or image folder is not found.
        """
        # Check label embeddings file
        if not os.path.isfile(label_embeddings_path):
            raise FileNotFoundError(f"Label embeddings file not found: {label_embeddings_path}")
        valid_pickle_extensions = ('.pkl', '.pickle')
        if not label_embeddings_path.lower().endswith(valid_pickle_extensions):
            raise ValueError("Provided label embeddings file must have a .pkl or .pickle extension.")
        
        # Check that data folder exists
        if not os.path.isdir(data_folder_path):
            raise NotADirectoryError(f"Data folder not found: {data_folder_path}")
        
        # Check that required files exist in the data folder
        required_files = [FILE_NAME, SIMPLIFIED_SYMBOL_FILE_NAME, LOCATIONS_FILE_NAME]
        for file_name in required_files:
            file_path = os.path.join(data_folder_path, file_name)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Required file not found in data folder: {file_path}")
        
        # Check that the image folder exists and is a directory
        image_folder = os.path.join(data_folder_path, IMAGE_FOLDER_NAME)
        if not os.path.isdir(image_folder):
            raise NotADirectoryError(f"Image folder not found: {image_folder}")

    def load_processed_data(self) -> pd.DataFrame:
        """
        Loads and processes the horizon data, including merging image and geographical data,
        filtering, imputing, encoding, and scaling features.
        
        Returns:
            pd.DataFrame: Processed horizon data.
        """
        df = self._load_and_preprocess_main_csv()
        df = self._merge_image_data(df)
        df = self._merge_geographical_data(df)
        df = self._filter_and_select_columns(df)
        df = self._impute_and_clean_data(df)
        df = self._process_target_column(df)
        df = self._encode_and_scale_features(df)
        df = self._onehot_encode_categorical_features(df)
        df = self._aggregate_data_for_sequential_training(df)
        return df
    
    def multi_label_stratified_shuffle_split(self, df: pd.DataFrame, n_splits : int = 1, train_val_test_frac: list[float] = [0.7, 0.15, 0.15], random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset according to the distribution of classes in categorical tabular features.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data to be split.
            n_splits (int): Number of re-shuffling & splitting iterations. Default is 1.
            train_val_test_frac (list[float]): Split ratios. Default is [0.7, 0.15, 0.15].
            random_state (int): Random seed. Default is None.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and test dataframes.
        """
        
        # Split dataset according to distribution of classes in categorical tabular features (including horizon labels)
        df_stratified_split_targets = df[self.stratified_split_targets]
        
        # First split training apart from validation and test
        ml_split_1 = MultilabelStratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=train_val_test_frac[1] + train_val_test_frac[2],
            random_state=random_state
        )
        for train_idx, val_and_test_idx in ml_split_1.split(df, df_stratified_split_targets):
            train_df, val_and_test_df = df.iloc[train_idx], df.iloc[val_and_test_idx]
            val_and_test_targets = df_stratified_split_targets.iloc[val_and_test_idx]
        
        # Second split validation and test
        ml_split_2 = MultilabelStratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=train_val_test_frac[2] / (train_val_test_frac[1] + train_val_test_frac[2]),
            random_state=random_state
        )
        for val_idx, test_idx in ml_split_2.split(val_and_test_df, val_and_test_targets):
            val_df, test_df = val_and_test_df.iloc[val_idx], val_and_test_df.iloc[test_idx]
        
        return train_df, val_df, test_df

    def _load_and_preprocess_main_csv(self) -> pd.DataFrame:
        """
        Loads and preprocesses the main CSV file containing horizon data.
        
        Returns:
            pd.DataFrame: Preprocessed horizon data.
        """
        df = pd.read_csv(os.path.join(self.data_folder_path, FILE_NAME))
        df = df.dropna(subset=['Horizontsymbol'])
        df_simple = pd.read_csv(os.path.join(self.data_folder_path, SIMPLIFIED_SYMBOL_FILE_NAME))
        df_simple.rename(
            columns={"relevanter Anteil = was sinntragend und detektierbar ist - es sind nicht alles gültige Symbole": "relevanter Anteil"}, 
            inplace=True
        )
        simplification_col = "relevanter Anteil"
        df['Horizontsymbol_relevant'] = df['Horizontsymbol'].apply(lambda x: HorizonDataProcessor._simplify_string(x, df_simple, simplification_col))
        cols = df.columns.tolist()
        cols.insert(cols.index('Horizontsymbol') + 1, 'Horizontsymbol_relevant')
        cols.pop()
        df = df[cols]
        return df

    def _merge_image_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges image data with the main horizon data.
        
        Args:
            df (pd.DataFrame): DataFrame containing the main horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with merged image data.
        """
        image_folder = os.path.join(self.data_folder_path, IMAGE_FOLDER_NAME)
        image_files = os.listdir(image_folder)
        img_files = pd.DataFrame(image_files, columns=['file'])
        img_files['Point'] = img_files['file'].str.split("_").map(lambda x: x[1]).astype(float)
        df = pd.merge(df, img_files, how='inner', on=['Point'])
        df['file'] = df['file'].map(lambda x: os.path.join(image_folder, x))
        return df

    def _merge_geographical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges geographical data with the main horizon data.
        
        Args:
            df (pd.DataFrame): DataFrame containing the main horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with merged geographical data.
        """
        df_loc = pd.read_csv(os.path.join(self.data_folder_path, LOCATIONS_FILE_NAME), encoding='unicode_escape')
        df_loc = df_loc.rename({'PointID': 'Point'}, axis=1)
        df_loc['Point'] = pd.to_numeric(df_loc['Point'], errors='coerce')
        df_loc = df_loc.dropna(subset=['Point'])
        df = pd.merge(df, df_loc, how='inner', on='Point')
        return df

    def _filter_and_select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and selects relevant columns from the horizon data.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with filtered and selected columns.
        """
        df = df[self.keep_columns]
        df['GrundwaStand'] = df['GrundwaStand'].str.replace('>', '').astype(float)
        df['ycoord'] = df['ycoord'].astype(float)
        df = df[df['Obergrenze'] < 100]
        return df

    def _impute_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values and cleans the horizon data.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with imputed and cleaned data.
        """
        df['Steine'] = df['Steine'].fillna(0.0)
        df['Durchwurzelung'] = df['Durchwurzelung'].fillna('Wf0').str.split(',').str[0]
        df['Exposition'] = df['Exposition'].replace(['0', '---'], 'KE')
        df['Woelbung'] = df['Woelbung'].replace(['0', '---'], 'GG')
        df['Reliefformtyp'] = df['Reliefformtyp'].replace(['0', '---'], 'H')
        df['LageImRelief'] = df['LageImRelief'].replace('0', 'Z')
        df['Karbonat'] = df['Karbonat'].replace('---', 'C0')
        df['GrundwaStufe'] = df['GrundwaStufe'].replace('---', '0')
        df.loc[df['Exposition'].isna() & (df['Neigung'] == 'N0'), 'Exposition'] = 'KE'
        df['Durchwurzelung'] = df['Durchwurzelung'].str.replace('f|g', '', regex=True)
        df['Karbonat'] = df['Karbonat'].str.replace(r'\.\d+', '', regex=True)
        df['GrundwaStufe'] = df['GrundwaStufe'].str.replace(r'\.\d+', '', regex=True)
        df['Neigung'] = df['Neigung'].str.replace(r'\.\d+', '', regex=True)
        df['Reliefformtyp'] = df['Reliefformtyp'].str[0]
        df['Hauptbodentyp'] = df['Hauptbodentyp'].str[:2]
        return df

    def _process_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the target column by mapping rare labels to frequent ones and encoding them.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with processed target column.
        """
        with open(self.label_embeddings_path, 'rb') as handle:
            emb_dict = pickle.load(handle)
        dict_mapping = {key.strip('.'): value for key, value in emb_dict['label2ind'].items()}
        df[self.target] = df[self.target].str.replace('+', '-', regex=False)
        df[self.target] = df[self.target].str.replace('°', '-', regex=False)
        df[self.target] = df[self.target].str.replace(r'\d+$', '', regex=True)
        rare_labels_mapping = {}
        list_dict_mapping = list(dict_mapping.keys())
        for lab in df[self.target].unique():
            if lab in list_dict_mapping:
                rare_labels_mapping[lab] = lab
            else:
                similarities = [levenshtein_distance(lab, freq_lab) for freq_lab in list_dict_mapping]
                best_match = list_dict_mapping[np.argmin(similarities)]
                rare_labels_mapping[lab] = best_match
        df[self.target] = df[self.target].map(rare_labels_mapping)
        df[self.target] = df[self.target].map(dict_mapping)
        df[self.target] = df[self.target].astype(int)
        return df

    def _encode_and_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical features and scales numerical features.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with encoded and scaled features.
        """
        for categ in self.categ_features:
            HorizonDataProcessor._encode_categorical_columns(df, categ)
        df = df.fillna(df.median(numeric_only=True))
        df = df.astype({'Bodenart': int, 'Bodenfarbe': int, 'Humusgehaltsklasse': int})
        scaler = MinMaxScaler()
        df[self.num_features] = scaler.fit_transform(df[self.num_features])
        return df
    
    def _onehot_encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encodes categorical features.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded categorical features.
        """
        geotemp_categ = list(set(self.categ_features).intersection(set(self.geotemp_img_infos)))
        df = pd.get_dummies(df, columns=geotemp_categ)
        self.geotemp_img_infos = [c for gt in self.geotemp_img_infos for c in df.columns if c.startswith(gt)]
        return df

    def _aggregate_data_for_sequential_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates data for sequential training by grouping and normalizing lists.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: Aggregated DataFrame for sequential training.
        """
        df = df.groupby('file', as_index=False).agg({
            **{col: 'first' for col in self.geotemp_img_infos},
            'Untergrenze': list,
            **{col: list for col in self.soil_infos},
            self.target: list
        }).reset_index()
        self.geotemp_img_infos = ['index'] + self.geotemp_img_infos
        df['Untergrenze'] = df['Untergrenze'].apply(HorizonDataProcessor._normalize_df_list)
        return df

    @staticmethod
    def _simplify_string(complex_string, mapping_df, col_name):
        """
        Simplifies a complex string based on the mapping.
        
        Args:
            complex_string (str): The complex string to be simplified.
            mapping_df (pd.DataFrame): DataFrame containing the mapping information.
            col_name (str): Column name in the mapping DataFrame.
        
        Returns:
            str: Simplified string.
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

    @staticmethod
    def _encode_categorical_columns(df, col_name):
        """
        Encodes categorical columns by replacing categories with numerical values.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            col_name (str): Name of the column to be encoded.
        """
        counts = df[col_name].value_counts()
        df[col_name] = df[col_name].replace(counts.index, range(len(counts)))
    
    @staticmethod
    def _normalize_df_list(lst, max_boundary=100.0):
        """
        Rounds the last list item to a max threshold and normalizes the whole list.
        
        Args:
            lst (list): List of numerical values.
            max_boundary (float): Maximum boundary value for normalization. Default is 100.0.
        
        Returns:
            list: Normalized list.
        """
        lst[-1] = max_boundary
        lst = [x/max_boundary for x in lst]
        return lst