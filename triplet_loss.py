import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss(model, X, y, margin=1.0):
    _, z, _, _ = model(X)
    unique_types = y.unique()
    num_types = unique_types.numel()

    #anomaly_prototypes = model.proto_layer.prototypes[-num_types:]
    anomaly_prototypes = model.proto_layer.prototypes[model.proto_layer.prototype_labels != 0]
    anomaly_proto_labels = model.proto_layer.prototype_labels_multi
    
    total_loss = 0.
    total_triplets = 0
    
    for i, anomaly_type in enumerate(anomaly_proto_labels):
        
        anchor = anomaly_prototypes[i]
        
        pos_mask = (y == anomaly_type)
        pos_samples = z[pos_mask]
        if pos_samples.shape[0] <= 0:
            continue
        
        neg_mask = (y != anomaly_type)
        neg_samples = z[neg_mask]
        
        pos_dists = torch.norm(anchor - pos_samples, dim=1)
        neg_dists = torch.norm(anchor - neg_samples, dim=1)
        
        for pos_dist in pos_dists:
            losses = F.relu(margin + pos_dist - neg_dists)
            total_loss += losses.sum()
            total_triplets += neg_dists.size(0)
        
    if total_triplets > 0:
        total_loss = total_loss / total_triplets
    
    return total_loss