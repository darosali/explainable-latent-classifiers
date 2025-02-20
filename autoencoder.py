import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

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
        """Get latent space representation"""
        return self.encoder(x)

def reconstruction_loss(x, x_hat):
    return nn.MSELoss()(x_hat, x)

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

