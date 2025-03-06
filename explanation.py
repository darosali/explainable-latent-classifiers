import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
import xgboost as xgb
import numpy as np

#def find_knn(X_train, y_train, k=9)

def show_importance(coefficients, feature_names, n_features=10):
    feature_importance = np.abs(coefficients)
    sorted_importance = np.argsort(feature_importance)[::-1]
    print(sorted_importance[:n_features])
    
    for imp_idx in sorted_importance[:n_features]:
        print(f"{feature_names[imp_idx]}: {coefficients[imp_idx]}")
        
    return sorted_importance[:n_features]

def show_importance_latent(coefficients, n_features=10):
    feature_importance = np.abs(coefficients)
    sorted_importance = np.argsort(feature_importance)[::-1]
    print(sorted_importance[:n_features])
    
    for imp_idx in sorted_importance[:n_features]:
        print(f"z_{imp_idx}: {coefficients[imp_idx]}")
        
    return sorted_importance[:n_features]