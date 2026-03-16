from pandas import DataFrame
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, List, Dict

class Dataset(TorchDataset):
    """Custom Torch Dataset for tabular data."""

    def __init__(self, df: DataFrame, target_col :str, apply_normalization: bool = False,
                 continuous_cols: List[str] = None, binary_cols: List[str] = None, 
                 categorical_cols: List[str] = None, column_groups: Dict[str, List[str]] = None,
                 group_categorical: bool = True, missing_threshold: int = 200):
        if target_col not in df.columns:
            raise ValueError(f'Target column {target_col} not found in data')
        
        df = df.copy()
        # Eliminate rows with missing values
        if missing_threshold is None:
            missing_threshold = 200
        features_cols = [col for col in df.columns if col != target_col]
        if features_cols:
            null_counts = df[features_cols].isnull().sum()
            features_to_drop = null_counts[null_counts >= missing_threshold].index
            print(f'Dropping {len(features_to_drop)} features with more than {missing_threshold} missing values')
            df = df.drop(columns=features_to_drop)
        else:
            print('No features to drop')
        df = df.dropna()
        y = df[target_col]
        X = df.drop(columns=[target_col])
        feature_cols = set(X.columns)

        self.continuous_cols = [col for col in (continuous_cols or []) if col in feature_cols]
        self.binary_cols = [col for col in (binary_cols or []) if col in feature_cols]
        self.categorical_cols = [col for col in (categorical_cols or []) if col in feature_cols]
        
       
        X_values = X.values
        self.scaler = None
        self.normalized_cols = []

        y_values = y.values
        print(f'Target range: [{y.min():.4f}, {y.max():.4f}]')

        self.features = torch.tensor(X_values, dtype=torch.float32)
        self.label = torch.tensor(y_values, dtype=torch.float32)
        self.feature_names = X.columns.tolist()
        
        self.group_categorical = group_categorical
        self.cat_sizes = []
        if not group_categorical:
            # Treat categorical columns as binary (each one-hot column is a binary variable)
            self.binary_cols = self.binary_cols + self.categorical_cols
            self.categorical_cols = []
        elif self.categorical_cols:
            # Group categorical variables
            if column_groups is None:
                raise ValueError('column_groups must be provided in config when group_categorical=True')
            
            for _, group_cols in column_groups.items():
                valid_cols = [col for col in group_cols if col in self.categorical_cols]
                if valid_cols:
                    self.cat_sizes.append(len(valid_cols))
        
        # Get indices for each variable type
        self.continuous_indices = []
        self.binary_indices = []
        self.categorical_indices = []
        
        continuous_set = set(self.continuous_cols)
        binary_set = set(self.binary_cols)
        categorical_set = set(self.categorical_cols)
        for i, col_name in enumerate(self.feature_names):
            if col_name in continuous_set:
                self.continuous_indices.append(i)
            elif col_name in binary_set:
                self.binary_indices.append(i)
            elif col_name in categorical_set:
                self.categorical_indices.append(i)
    
    def get_variable_splits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split input tensor by variable type.
        Returns: (continuous, binary, categorical)
        """
        cont = x[:, self.continuous_indices] if self.continuous_indices else torch.empty(x.shape[0], 0, device=x.device)
        bin_ = x[:, self.binary_indices] if self.binary_indices else torch.empty(x.shape[0], 0, device=x.device)
        cat = x[:, self.categorical_indices] if self.categorical_indices else torch.empty(x.shape[0], 0, device=x.device)
        return cont, bin_, cat


    def __len__(self)-> int:
        return len(self.label)
    
    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.label[idx]
        return feature, label
    
    @property
    def n_continuous(self) -> int:
        return len(self.continuous_indices)
    
    @property
    def n_binary(self) -> int:
        return len(self.binary_indices)
    
    @property
    def n_categorical(self) -> int:
        return sum(self.cat_sizes) if self.cat_sizes else 0
    
    @property
    def n_categorical_vars(self) -> int:
        return len(self.cat_sizes)
