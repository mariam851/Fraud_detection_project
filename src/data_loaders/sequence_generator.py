# src/data_loaders/sequence_generator.py
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class SequenceGenerator:
    """
    Prepares Hybrid (Temporal + Structural) features for LSTM training.
    Optimized for high-imbalance datasets using Stratified Splitting.
    """
    def __init__(self, features_list, batch_size=2048):
        self.features_list = features_list
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def prepare_data(self, df, target_col='isFraud', test_size=0.2, apply_sampling=True):
        if apply_sampling:
            df_fraud = df[df[target_col] == 1]
            df_normal = df[df[target_col] == 0].sample(frac=0.2, random_state=42)
            df = pd.concat([df_fraud, df_normal]).sort_values(by='step')

        X = df[self.features_list].values
        y = df[target_col].values

        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, "models/scaler.pkl")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_t = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(1)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
        
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1)
        y_test_t = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=self.batch_size)

        return train_loader, test_loader, y_test
