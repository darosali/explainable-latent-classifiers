import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm
import xgboost as xgb
import numpy as np

def find_knn(x, X_train_n, y_train_n, X_train_a, y_train_a, k=9):
    knn_normal = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    knn_normal.fit(X_train_n)
    distances_n, indices_n = knn_normal.kneighbors(x, n_neighbors=k)
    print(indices_n)
    print(distances_n)
    
    knn_anomaly = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    knn_anomaly.fit(X_train_a)
    distances_a, indices_a = knn_anomaly.kneighbors(x, n_neighbors=k)
    print(indices_a)
    print(distances_a)

    lasso_x_train = np.vstack((X_train_n[indices_n.flatten().tolist()], X_train_a[indices_a.flatten().tolist()]))
    lasso_y_train = np.concatenate((y_train_n[indices_n.flatten().tolist()], y_train_a[indices_a.flatten().tolist()]))
    print(lasso_x_train.shape)
    print(lasso_y_train.shape)

    return lasso_x_train, lasso_y_train

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
