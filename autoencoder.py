import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
import xgboost as xgb

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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, activation=nn.Tanh):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_layers, latent_dim, activation)
        self.decoder = Decoder(input_dim, hidden_layers, latent_dim, activation)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
        
    def encode(self, x):
        return self.encoder(x)

class Classifier(nn.Module):
    """Classifier that predicts labels from latent space"""
    def __init__(self, latent_dim, num_classes, hidden_layers=[8], activation=nn.ReLU):
        super(Classifier, self).__init__()
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes)) 
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)

class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, num_classes, activation=nn.Tanh):
        super(CombinedModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_layers, latent_dim, activation)
        self.decoder = Decoder(input_dim, hidden_layers, latent_dim, activation)
        self.classifier = Classifier(latent_dim, num_classes)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_pred = self.classifier(z)
        return x_hat, y_pred, z

def combined_loss(x, x_hat, y_pred, y_true, alpha=0.5, class_weight=None):
    recon_loss = nn.MSELoss()(x_hat, x)  # reconstruction loss
    class_loss = nn.CrossEntropyLoss(weight=class_weight)(y_pred, y_true)  # classification loss
    return alpha * recon_loss + (1 - alpha) * class_loss

def reconstruction_loss(x, x_hat):
    return nn.MSELoss()(x_hat, x)

def classification_loss(y_pred, y_true, class_weights=None):
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    return loss_fn(y_pred, y_true)

def train(train_loader, val_loader, input_dim, hidden_layers, latent_dim, epochs=10, lr=0.001):
    autoencoder = Autoencoder(input_dim, hidden_layers, latent_dim)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=0.001)
    
    for epoch in range(epochs):
        autoencoder.train()
        total_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for x, _ in train_loader:
            x_hat = autoencoder(x)
            loss = reconstruction_loss(x, x_hat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
            train_progress.update(1)
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loss
        autoencoder.eval()
        total_val_loss = 0.0
        #val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
        with torch.no_grad():
            for x, _  in val_loader:
                x_hat = autoencoder(x)
                loss = reconstruction_loss(x, x_hat)
                total_val_loss += loss.item()
                #val_progress.set_postfix(loss=loss.item())
                #val_progress.update(1)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return autoencoder

def train_classifier(latent_train, y_train, latent_val, y_val, latent_dim, num_classes, epochs=10, lr=0.001, class_weights=None):
    train_dataset = TensorDataset(latent_train, y_train)
    val_dataset = TensorDataset(latent_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    classifier = Classifier(latent_dim, num_classes)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(epochs):
        classifier.train()
        total_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training Classifier]")
        
        for z, y in train_progress:
            y_pred = classifier(z)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loss
        classifier.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for z, y in val_loader:
                y_pred = classifier(z)
                loss = criterion(y_pred, y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Class Loss = {avg_train_loss:.4f}, Val Class Loss = {avg_val_loss:.4f}")
    
    return classifier

def train_xgboost_classifier(latent_train, y_train, latent_val, y_val):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    
    model.fit(latent_train, y_train, eval_set=[(latent_val, y_val)], verbose=True)
    return model

def train_combined(train_loader, val_loader, input_dim, hidden_layers, latent_dim, num_classes, epochs=10, lr=0.001, alpha=0.5, class_weights=None):
    model = CombinedModel(input_dim, hidden_layers, latent_dim, num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for x, y in train_progress:
            x_hat, y_pred, _ = model(x)
            loss = combined_loss(x, x_hat, y_pred, y, alpha, class_weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x_hat, y_pred, _ = model(x)
                loss = combined_loss(x, x_hat, y_pred, y, alpha, class_weights)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return model
