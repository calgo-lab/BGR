from typing import Tuple
from jellyfish import levenshtein_distance
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning) # silence FutureWarning for downcasting in 'replace'; must come before importing pandas
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' to silence SettingWithCopyWarning

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
        # self.soil_infos must have the same order as in the dataset, or the concatenation of tab predictions in end2end model will be wrong
        self.soil_infos = ['Steine', 'Bodenart', 'Bodenfarbe', 'Karbonat', 'Humusgehaltsklasse', 'Durchwurzelung']
        self.tabulars_output_dim_dict = {key: None for key in self.soil_infos}
        self.geotemp_img_infos = [
            'Probenahme_Monat', 'Probenahme_Jahr', 'xcoord', 'ycoord', 'Bodenklimaraum_Name',
            'Landnutzung', 'BZE_Moor', 'Hauptbodentyp', 'GrundwaStufe', 'GrundwaStand',
            'Moormaechtigkeit', 'Torfmaechtigkeit', 'Neigung', 'Exposition', 'Woelbung', 
            'Reliefformtyp', 'LageImRelief', 'KV_0_30', 'KV_30_100', 'file'
        ]
        # Note: we don't need to stratify wrt stones (it's numerical)
        self.stratified_split_targets = ['Bodenart', 'Bodenfarbe', 'Karbonat', 'Humusgehaltsklasse', 'Durchwurzelung', 'Horizontsymbol_relevant']
        self.embeddings_dict = None
        
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
        df = self._process_soil_type_column(df)
        df = self._process_soil_color_column(df)
        df = self._process_target_column(df)
        df = self._encode_and_scale_features(df)
        df = self._onehot_encode_categorical_features(df)
        df = self._aggregate_data_for_sequential_training(df)
        
        # Set the number of classes / outputs for each tabular feature
        for key in self.tabulars_output_dim_dict.keys():
            if key == 'Steine':
                self.tabulars_output_dim_dict[key] = 1
            else:
                self.tabulars_output_dim_dict[key] = max(df[key].apply(max)) + 1
        
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
        df = df.dropna(subset=['Horizontsymbol']) # There is one useless row full of NaNs
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

    def _process_soil_type_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the soil type column by simplifying the labels according to the rules in the book (pages 142-159).
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with processed soil type column.
        """
        # Summarize the main soil types
        mapping_main = {"Uls": "lu", "Ss": "ss", "fSgs": "ss", "fS": "ss", "fSms": "ss", "mSfs": "ss", 
                        "mS": "ss", "mSgs": "ss", "gS": "ss", "Su2": "ls", "Su3": "us", "Su4": "us", 
                        "Slu": "sl", "Sl2": "ls", "Sl3": "ls", "Sl4": "sl", "St2": "ls", "St3": "sl", 
                        "Uu": "su", "Us": "su", "Ut2": "lu", "Ut3": "lu", "Ut4": "tu", "Ls2": "ll", 
                        "Ls3": "ll", "Ls4": "ll", "Lu": "tu", "Lt2": "ll", "Lt3": "ut", "Lts": "tl", 
                        "Ts2": "lt", "Ts3": "tl", "Ts4": "tl", "Tu4": "ut", "Tu3": "ut", "Tu2": "lt", 
                        "Tl": "lt", "Tt": "lt", "fSu2": "ls", "fSu3": "us", "fSu4": "us", "fSlu": "sl", 
                        "fSl2": "ls", "fSl3": "ls", "fSl4": "sl", "fSt2": "ls", "fSt3": "sl", "mSu2": "ls", 
                        "mSu3": "us", "mSu4": "us", "mSlu": "sl", "mSl2": "ls", "mSl3": "ls", "mSl4": "sl", 
                        "mSt2": "ls", "mSt3": "sl", "gSu2": "ls", "gSu3": "us", "gSu4": "us", "gSlu": "sl", 
                        "gSl2": "ls", "gSl3": "ls", "gSl4": "sl", "gSt2": "ls", "gSt3": "sl"}
        
        # Summarize the F-class
        mapping_f = {"Fmt": "Fm", "Fmu": "Fm", "Fm": "Fm", "Fms": "Fm", "Fh": "Fh", 
                     "Fhl": "Fh", "Fhh": "Fh", "Fhg": "Fh", "F": "F", "Fmk": "Fm", "Fmi": "Fm"}
        
        # Summarize the H-class
        mapping_h = {"H": "H", "Ha": "Ha", "Hh": "Hh", "Hha": "Hh", "Hhe": "Hh", "Hhi": "Hh", 
                     "Hhk": "Hh", "Hhs": "Hh", "Hhsa": "Hh", "Hhsu": "Hh", "Hhsy": "Hh", "Hn": "Hn", 
                     "Hnb": "Hn", "Hnd": "Hn", "Hnle": "Hn", "Hnmy": "Hn", "Hnp": "Hn", "Hnq": "Hn", 
                     "Hnr": "Hn", "Hu": "Hu", "Hulb": "Hu", "Hulk": "Hu"}
        
        # Summarize rare soil types (classified according to extra finer grained rules)
        # Rule: replace them with the soil type in the horizon above them. They are thin and at the bottom anyway.
        mapping_rare = {'k': 'ls', 'z': 'll', 'v': 'tl'}
        
        df['Bodenart'] = df['Bodenart'].replace(mapping_main).replace(mapping_f).replace(mapping_h).replace(mapping_rare)
        return df
    
    def _process_soil_color_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the soil color column by trimming inconsistent labels and clustering the remaining ones.
        
        Args:
            df (pd.DataFrame): DataFrame containing the horizon data.
        
        Returns:
            pd.DataFrame: DataFrame with processed soil color column.
        """
        
        # Some GLEY values have inconsistent Munsell notation. Fix them.
        df['Bodenfarbe'] = df['Bodenfarbe'].str.replace(r'_/1|_/2', '', regex=True)
        # Remove prefix 'WHITE' (just extra marker for Munsell values above 8)
        df['Bodenfarbe'] = df['Bodenfarbe'].str.replace('WHITE ', '', regex=False)
        # Grid clustering based on Munsell value and chroma
        df['Bodenfarbe'] = df['Bodenfarbe'].map(HorizonDataProcessor._cluster_soil_color)
        
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
            self.embeddings_dict = pickle.load(handle)
        
        # For HCE: create separate dict where labels made out only of main symbols are stripped of the full stop '.'
        # Note: We will use the label indexes from the embedding dictionary instead of a new sparse one-hot encoding to ease access 
        # to the embedding vectors during training via the original indexes from the emb_dict
        dict_mapping = {key.strip('.'): value for key, value in self.embeddings_dict['label2ind'].items()}
        
        # Replace '+' with '-' in the target column (see Label_Graph.ipynb)
        df[self.target] = df[self.target].str.replace('+', '-', regex=False)
        df[self.target] = df[self.target].str.replace('°', '-', regex=False) # also these, so that there is only one type of mixtures (the minus-mixture)
        # Remove trailing numbers from the labels in the target column (they only account for how often the horizon is seen in the same picture)
        df[self.target] = df[self.target].str.replace(r'\d+$', '', regex=True)
        
        # Replace (rare) main symbols that do not have a counterpart in the graph
        # Note: Mapping was provided by domain experts
        # Note: L-Horizons are not relevant and do not occur at all in the preprocessed dataframe
        df[self.target] = df[self.target].str.replace('F', 'H', regex=False) # because of common Moor
        df[self.target] = df[self.target].str.replace('O', 'H', regex=False) # because both organic
        df[self.target] = df[self.target].str.replace('T', 'P', regex=False) # because both rich in tone
        df[self.target] = df[self.target].str.replace('Y', 'G', regex=False) # because both dry soils with similar humid features
        
        # Map rare labels to frequent labels via Levenshtein distance
        rare_labels_mapping = {}
        list_dict_mapping = list(dict_mapping.keys())
        for lab in df[self.target].unique():
            if lab in list_dict_mapping:
                rare_labels_mapping[lab] = lab # for applying the map later in the df, we need the identity mappings as well
            else:
                similarities = [levenshtein_distance(lab, freq_lab) for freq_lab in list_dict_mapping]
                best_match = list_dict_mapping[np.argmin(similarities)]
                rare_labels_mapping[lab] = best_match
                
        # Correct imprecise mappings (some levenshtein results are not geologically plausible)
        for lab in rare_labels_mapping:
            # Skip if the label is already in graph_labels
            if lab in dict_mapping:
                continue
            
            # Get the components of mixtures or the whole label if non-mixture
            lab_splits = lab.split('-')
            mapped_labels = []
            for ls in lab_splits:
                if ls in dict_mapping:
                    mapped_labels.append(ls)
            # If the mixture has multiple components in the graph, assign the last component as the frequent label
            # (Analogy German: Hausschlüssel is a Schlüssel)
            if len(mapped_labels) > 0:
                rare_labels_mapping[lab] = mapped_labels[-1]        
                
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
        
        # Scale numerical features
        # Note: We don't scale the 'Steine' feature, with MinMax due to its skewed distribution
        # We scale it to 0-10, so that most of the values are in the range of 0-1
        df['Steine'] = df['Steine'] / 10.0
        
        features_to_minmax_scale = [feature for feature in self.num_features if feature != 'Steine']
        scaler = MinMaxScaler()
        df[features_to_minmax_scale] = scaler.fit_transform(df[features_to_minmax_scale])
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
    
    @staticmethod
    def _cluster_soil_color(color_label):
        """
        Clusters the soil color label into grid groups based on Munsell notation.
        
        Args:
            color_label (str): The soil color label in Munsell notation.
            
        Returns:
            str: Clustered soil color label.
        """
        try:
            # Split the value into parts
            page, value_chroma = color_label.split(' ')
            value, chroma = value_chroma.split('/')
            
            # Group the first number
            if value in ['2.5', '3']:
                first_group = '<=3'
            elif value in ['4', '5']:
                first_group = '4-5'
            elif value in ['6', '7']:
                first_group = '6-7'
            else:
                first_group = '>=8'
            
            # Group the second number (the GLEY pages need to be treated differently)
            if page == 'GLEY1':
                if chroma in ['N', '10Y']:
                    first_group = 'N-10Y'
                elif chroma in ['5GY', '10GY']:
                    first_group = '5GY-10GY'
            elif page == 'GLEY2':
                if chroma in ['10G', '5BG']:
                    first_group = '10G-5BG'
                elif chroma in ['10BG', '5B']:
                    first_group = '10BG-5B'
                elif chroma in ['10B', '5PB']:
                    first_group = '10B-5PB'
            else:
                if chroma in ['1', '2']:
                    second_group = '1-2'
                elif chroma in ['3', '4']:
                    second_group = '3-4'
                elif chroma in ['6', '8']:
                    second_group = '6-8'
            
            return f"{page} {first_group}/{second_group}"
        except:
            return np.nan  # Handle invalid or missing values