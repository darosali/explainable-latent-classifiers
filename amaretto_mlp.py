import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import polars as pl
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils_b import *


class MLP(nn.Module):
    def __init__(self, input_size, hidden_units=[64, 64], output_size=6, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_units[0]))
        layers.append(nn.BatchNorm1d(hidden_units[0]))
        layers.append(activation_fn)

        # Hidden layers
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_units[i + 1]))
            layers.append(activation_fn)

        # Output layer
        layers.append(nn.Linear(hidden_units[-1], output_size))


        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        # If it's a multi-class classification, use argmax to get the predicted class
        with torch.no_grad():
            output = self.forward(X)
            return torch.argmax(output, dim=1).cpu().numpy()

class AmarettoDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: numpy array of shape (num_samples, num_features)
        labels: numpy array of shape (num_samples,) or (num_samples, 1)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_data(filename):
    df = pl.scan_parquet(filename).collect()
    return df

def split_data(df):
    #df_train = df.filter((pl.col('datetime') < datetime(2019, 3, 11, 0, 0, 0)))
    #df_val = df.filter((pl.col('datetime') >= datetime(2019, 3, 11, 0, 0, 0)) & (pl.col('datetime') < datetime(2019, 3, 18, 0, 0, 0)))
    #df_test = df.filter((pl.col('datetime') >= datetime(2019, 3, 18, 0, 0, 0)))
    df_train = df.filter(df['datetime'] < datetime(2019, 1, 7, 0, 0, 0))
    df_val = df.filter((df['datetime'] >= datetime(2019, 1, 7, 0, 0, 0)) & (df['datetime'] < datetime(2019, 1, 8, 0, 0, 0)))
    df_test = df.filter((df['datetime'] >= datetime(2019, 1, 7, 0, 0, 0)) & (df['datetime'] < datetime(2019, 1, 8, 0, 0, 0)))
    return df_train, df_val, df_test

def get_args():
    parser = argparse.ArgumentParser(description="Fraud detection MLP model.")

    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[16, 16], help="Sizes of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer.")
    parser.add_argument("--activation_fn", type=str, default="ReLU", choices=["ReLU", "ELU", "Tanh", "LeakyReLU"],
                        help="Activation function (default: ReLU).")

    return vars(parser.parse_args())

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)
    
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        # Follow progress
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Eval after each epoch
        evaluate_performance_m(model, train_loader, val_loader, "performance1.txt")

def evaluate_performance_m(model, train_loader, test_loader, output_file = 'output/performance_evaluation_multiclass.txt'):
    output_lines = []

    def evaluate_predictions(loader, label):
        # Get predictions
        #num_batches = len(loader)
        #total_rows = sum(len(batch[0]) for batch in loader)
        #print(f"Number of batches: {num_batches}, Total rows processed: {total_rows}")
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for features, labels in loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        
        # Calculate class-specific metrics
        class_precision = precision_score(y_true, y_pred, average=None)
        class_recall = recall_score(y_true, y_pred, average=None)
        class_f1 = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate output string
        results = (
            f"{label} Results:\n"
            f"Confusion Matrix:\n{cm}\n"
            f"Overall Accuracy: {accuracy:.4f}\n"
            f"Precision (Weighted): {precision_weighted:.4f}\n"
            f"Recall (Weighted): {recall_weighted:.4f}\n"
            f"F1 Score (Weighted): {f1_weighted:.4f}\n"
        )
        
        # Add class-specific metrics to the results
        results += "Class-Specific Metrics:\n"
        for i, (prec, rec, f1) in enumerate(zip(class_precision, class_recall, class_f1)):
            results += (
                f"Class {i} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}\n"
            )
        
        return results, y_true, y_pred

    # Evaluate on train and test data
    #train_results, _, _ = evaluate_predictions(train_loader, "Train")
    test_results, test_y_true, test_y_pred = evaluate_predictions(test_loader, "Test")

    # Append results for output
    #output_lines.append(train_results)
    output_lines.append(test_results)

    # Save results to a file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    # Print results
    for line in output_lines:
        print(line)

    return test_y_true, test_y_pred

if __name__ == '__main__':

    args = get_args()

    ACTIVATION_FUNCTIONS = {
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Tanh": torch.nn.Tanh(),
        "LeakyReLU": torch.nn.LeakyReLU(),
    }
    activation_fn = ACTIVATION_FUNCTIONS[args["activation_fn"]]

    # Load already transformed and scaled dataset
    df = load_data(r'amaretto_transformed_scaled.pq')
    #df.write_csv("amaretto_transformed_scaled.csv")
    df_train, df_val, df_test = split_data(df)
    columns_to_drop = ['id', 'Anomaly', 'Anomaly_bin', 'Originator', 'datetime']

    target_column = 'Anomaly'
    y_train = df_train.select(target_column).to_numpy().flatten()
    df_train = df_train.drop(columns_to_drop)
    X_train = df_train.to_numpy()
    y_val = df_val.select(target_column).to_numpy().flatten()
    X_val = df_val.drop(columns_to_drop).to_numpy()
    y_test = df_test.select(target_column).to_numpy().flatten()
    X_test = df_test.drop(columns_to_drop).to_numpy()
    print(X_train.shape)

    train_dataset = AmarettoDataset(X_train, y_train)
    val_dataset = AmarettoDataset(X_val, y_val)
    test_dataset = AmarettoDataset(X_test, y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights[3] *= 10
    print(class_weights)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # Manually defining class weights
    # class_weights = {0: 0.7, 1: 0.02, 2: 0.02, 3: 0.05, 4: 0.10, 5: 0.11}
    # sample_weights = np.array([class_weights[int(label)] for label in y_train.flatten()])
    # sample_weights = torch.from_numpy(sample_weights).float()
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # sampler = CustomWeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    #train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=2)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2)
    print("Data has been loaded")

    input_size = X_train.shape[1]
    hidden_layers = args["hidden_layers"]
    model = MLP(input_size=input_size, hidden_units=hidden_layers, activation_fn=activation_fn)
    #model.load_state_dict(torch.load('weights_mlp_amaretto2.pth', map_location=torch.device('cpu')))

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    #print("Training model")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=args["epochs"])
    print("Model eval")
    model.eval()
    y_true, y_pred = evaluate_performance_m(model, train_loader, test_loader, "performance_mlp_small.txt")
    #torch.save(model.state_dict(), "weights_mlp_amaretto2.pth")
    
    # Get plots and SHAP explanations
    
    #save_confusion_matrix(y_test, y_pred)
    #print("Model output shape:", model(test_batch).shape)  # Example batch
    #X_shap_tr, y_shap_tr = sample_shap_data(X_train, y_train, n_samples=400, anomaly_ratio=0.3)
    #X_shap_test, y_shap_test = sample_shap_data(X_test, y_test, n_samples=1000, anomaly_ratio=0.3)
    #print(X_shap_test.shape)
    #print(X_test.shape)
    #print(X_shap_tr.shape)
    #print(X_shap_test.shape)
    # print(len(train_df.columns))
    #get_shap_explanations_mlp(model, X_shap_tr, X_shap_test, df_train.columns)
