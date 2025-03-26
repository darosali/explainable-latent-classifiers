import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import numpy as np

class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, latent_dim, num_classes):
        super(PrototypeLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        self.prototype_labels = torch.arange(num_prototypes) % num_classes
        
    def forward(self, x):
        distances = torch.cdist(x, self.prototypes)
        similarity_scores = torch.log(1 + distances)/distances
        return similarity_scores

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes, num_classes, latent_dim, init_weights=True):
        super(ProtoPNet, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        
        self.proto_layer = PrototypeLayer(num_prototypes, latent_dim, num_classes)
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=False)
        
        if init_weights:
            self._initialize_weights()
            
    def _initialize_weights(self):
        
        with torch.no_grad():
            self.last_layer.weight.zero_()
            
            for j in range(self.num_prototypes):
                prototype_class = self.proto_layer.prototype_labels[j]
                self.last_layer.weight[prototype_class, j] = 1.0
                
                for k in range(self.num_classes):
                    if k != prototype_class:
                        self.last_layer.weight[k, j] = -0.5
    
    def forward(self, x):
        similarity_scores = self.proto_layer(x)
        logits = self.last_layer(similarity_scores)
        return logits
                
        
def proto_loss(y_true, prototype_distances, z, prototype_labels, lambda_clst=0.8, lambda_sep=0.08):
    correct_class_mask = y_true.unsqueeze(1) == prototype_labels.unsqueeze(0)
    clst_distances = prototype_distances.clone()
    clst_distances[~correct_class_mask] = float('inf')
    L_clst = torch.mean(torch.min(clst_distances, dim=1)[0])
    
    incorrect_class_mask = y_true.unsqueeze(1) != prototype_labels.unsqueeze(0)
    sep_distances = prototype_distances.clone()
    sep_distances[~incorrect_class_mask] = float('inf')
    L_sep = torch.mean(torch.min(sep_distances, dim=1)[0])
    
    return lambda_clst * L_clst + lambda_sep * L_sep
    
if __name__ == "__main__":
    pass