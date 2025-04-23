import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import numpy as np
from triplet_loss import *


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, activation=nn.Tanh):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            #layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, activation=nn.Tanh):
        super(Decoder, self).__init__()
        layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            #layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)
    
class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, num_neg, latent_dim, num_classes):
        super(PrototypeLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        #self.prototype_labels = torch.arange(num_prototypes) % num_classes
        self.prototype_labels = torch.cat([
            torch.zeros(num_neg, dtype=torch.long),
            torch.ones(num_prototypes - num_neg, dtype=torch.long)
        ])
        labels = torch.arange(1, 6)
        self.prototype_labels_multi = labels.repeat_interleave((num_prototypes - num_neg) // 5)
        
    def forward(self, x):
        distances = torch.cdist(x, self.prototypes)
        return distances

class ProtoPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_prototypes, num_neg, num_classes, latent_dim, activation = nn.Tanh, init_weights=True):
        super(ProtoPNet, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        self.encoder = Encoder(input_dim, hidden_layers, latent_dim, activation)
        self.decoder = Decoder(input_dim, hidden_layers, latent_dim, activation)
        self.proto_layer = PrototypeLayer(num_prototypes, num_neg, latent_dim, num_classes)
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
        z = self.encoder(x)
        x_hat = self.decoder(z)
        distances = self.proto_layer(z)
        similarity_scores = torch.log((distances + 1) / (distances + self.epsilon))
        #similarity_scores = torch.log(1 + distances)
        logits = self.last_layer(similarity_scores)
        return x_hat, z, distances, logits
                

def proto_loss(y_pred, y_true, x_hat, x, prototype_distances, prototype_labels, class_weights=None, lambda_clst=0.8, lambda_sep=0.08, alpha=0.5):
    
    # Cross-entropy classification loss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    class_loss = loss_fn(y_pred, y_true)
    recon_loss = nn.MSELoss()(x_hat, x)

    # Encourages each sample to be close to at least one prototype of the same class.
    # For each sample, find the prototype of its correct class that is closest.
    correct_class_mask = y_true.unsqueeze(1) == prototype_labels.unsqueeze(0)
    clst_distances = prototype_distances.clone()
    clst_distances[~correct_class_mask] = float('inf')
    # Compute Clst loss as the minimum distance to any correct-class prototype
    L_clst = torch.mean(torch.min(clst_distances, dim=1)[0])

    # Encourages each sample to be far from prototypes of other classes.
    # For each sample, find the closest prototype belonging to a different class.
    incorrect_class_mask = y_true.unsqueeze(1) != prototype_labels.unsqueeze(0)
    sep_distances = prototype_distances.clone()
    sep_distances[~incorrect_class_mask] = float('inf')
    L_sep = -torch.mean(torch.min(sep_distances, dim=1)[0])
    
    return alpha * class_loss + (1. - alpha) * recon_loss + lambda_clst * L_clst + lambda_sep * L_sep

def push(model, train_loader):
    
    model.eval()
    num_prototypes = model.num_prototypes

    # Track the minimum distance for each prototype
    global_min_proto_dist = torch.full((num_prototypes,), float('inf'))
    # Store the best feature vectors that will replace the prototypes
    best_prototypes = model.proto_layer.prototypes.clone()
    
    with torch.no_grad():
        for x, y in train_loader:
            _, z, _, _ = model(x)
            
            for j, p in enumerate(model.proto_layer.prototypes):
                # Get the class of the current prototype
                prototype_class = model.proto_layer.prototype_labels[j]
                # Select only samples of the same Class
                class_mask = y == prototype_class
                class_latents = z[class_mask]
                
                if len(class_latents) > 0:
                    # Compute L2 distances between prototype `p` and all class-specific feature vectors `zi`
                    distances = torch.cdist(p.unsqueeze(0), class_latents)
                    # Find the closest feature vector in the batch
                    batch_min_dist, batch_closest_idx = torch.min(distances, dim=1)
                    
                    # Update the prototype if we found a closer match
                    if batch_min_dist < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = batch_min_dist.item()
                        best_prototypes[j] = class_latents[batch_closest_idx].squeeze(0)
                        
    model.proto_layer.prototypes.data.copy_(best_prototypes)

def train_last_layer(model:ProtoPNet, train_loader, lr=0.0001, epochs=1, class_weights=None):
    
    model.eval()
    model.last_layer.train()
    
    optimizer = optim.AdamW(model.last_layer.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:

            with torch.no_grad():  # Freeze encoder & prototype layer
                _, z, _, _ = model(x)
                distances = model.proto_layer(z)
                similarity_scores = torch.log((distances + 1) / (distances + 1e-4))

            logits = model.last_layer(similarity_scores)  # Compute logits
            loss = loss_fn(logits, y)  # Compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
    

def train_protopnet(model, train_loader, val_loader, epochs, lr=0.001, class_weights=None, lambda_clst=0.8, lambda_sep=0.08):
    print(class_weights)
    # Optimizer for convolutional layers (except last layer)
    feature_optimizer = optim.AdamW(list(model.encoder.parameters()) + list(model.proto_layer.parameters()), lr=lr)
    best_f1 = 0.0 
    best_model_weights = None
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # --- Phase 1: Stochastic Gradient Descent (SGD) ---
        model.train()
        total_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for x, y in train_loader:
            feature_optimizer.zero_grad()

            x_hat, z, prototype_distances, logits = model(x)
            #loss = proto_loss_v(model, logits, y, prototype_distances, class_weights=class_weights)
            loss = proto_loss(logits, y, x_hat, x, prototype_distances, model.proto_layer.prototype_labels, lambda_clst=lambda_clst, lambda_sep=lambda_sep)

            loss.backward()
            feature_optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
            train_progress.update(1)

        print(f"Phase 1 - Training Loss: {total_loss / len(train_loader):.4f}")

        # --- Phase 2: Prototype Projection ---
        model.eval()
        push(model, train_loader)
        print("Phase 2 - Prototype Projection Completed")

        # --- Phase 3: Last Layer Optimization ---
        train_last_layer(model, train_loader, lr=lr, epochs=1, class_weights=class_weights)
        print("Phase 3 - Last Layer Optimization Completed")

        # Evaluate on validation set
        if val_loader:
            best_f1, best_model_weights = evaluate(model, val_loader, epoch, best_f1, best_model_weights, class_weights=class_weights)

def train_protopnet_triplet(model, train_loader, val_loader, epochs, X_triplet, y_triplet, lr=0.001, class_weights=None, lambda_triplet=0.5, lambda_clst=0.8, lambda_sep=0.08, filepath="model_ppnet_best_triplet.pth"):
    print(lambda_triplet)
    # Optimizer for convolutional layers (except last layer)
    feature_optimizer = optim.AdamW(list(model.encoder.parameters()) + list(model.proto_layer.parameters()), lr=lr)
    best_f1 = 0.0 
    best_model_weights = None
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # --- Phase 1: Stochastic Gradient Descent (SGD) ---
        model.train()
        total_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for x, y in train_loader:
            feature_optimizer.zero_grad()

            x_hat, z, prototype_distances, logits = model(x)
            #loss = proto_loss_v(model, logits, y, prototype_distances, class_weights=class_weights)
            loss = proto_loss(logits, y, x_hat, x, prototype_distances, model.proto_layer.prototype_labels, lambda_clst=lambda_clst, lambda_sep=lambda_sep)
            t_loss = triplet_loss(model, X_triplet, y_triplet)
            loss += lambda_triplet * t_loss

            loss.backward()
            feature_optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
            train_progress.update(1)

        print(f"Phase 1 - Training Loss: {total_loss / len(train_loader):.4f}")

        # --- Phase 2: Prototype Projection ---
        model.eval()
        push(model, train_loader)
        print("Phase 2 - Prototype Projection Completed")

        # --- Phase 3: Last Layer Optimization ---
        train_last_layer(model, train_loader, lr=lr, epochs=1, class_weights=class_weights)
        print("Phase 3 - Last Layer Optimization Completed")

        # Evaluate on validation set
        if val_loader:
            best_f1, best_model_weights = evaluate(model, val_loader, epoch, best_f1, best_model_weights, class_weights=class_weights, filepath=filepath)

def evaluate(model, val_loader, epoch, best_f1, best_model_weights, class_weights=None, filepath="model_ppnet_triplet_xor.pth"):

    model.eval()
    y_true = []
    y_pred = []
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    total_val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            _, _, _, logits = model(x)
            y_pred_batch = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_batch.cpu().numpy())
            loss = loss_fn(logits, y)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
    print(classification_report(y_true, y_pred))

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, filepath)
        print(f"New best model saved with Macro F1 = {macro_f1:.4f}")

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        
    return best_f1, best_model_weights


if __name__ == "__main__":
    pass