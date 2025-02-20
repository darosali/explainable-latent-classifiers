import numpy as np
#import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, roc_curve, auc, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import shap
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler

def undersample(X, y):
    desired_proportion = 0.3
    fraud_samples = (y==1).sum()
    total_samples = int(fraud_samples / desired_proportion)
    undersampler = RandomUnderSampler(sampling_strategy={0: total_samples - fraud_samples, 1: fraud_samples}, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    return X_resampled, y_resampled

def oversample(X, y):
    desired_proportion = 0.3
    legit_samples = (y == 0).sum()
    fraud_samples = int(legit_samples * desired_proportion)
    smote = SMOTE(sampling_strategy={0: legit_samples, 1: fraud_samples}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)


def show_importance(importance, features, filename='output/feature_importances_xgb_m.txt'):
    output_lines = []
    # Create a list of (feature_name, importance_value) tuples
    feature_importances = [(feature, importance.get(f"f{i}", 0)) for i, feature in enumerate(features)]
    # Sort the list by importance values in descending order
    feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Store the sorted feature importances
    output_lines = [f"{feature}: {importance_value:.4f}\n" for feature, importance_value in feature_importances_sorted]

    # Write feature importances to a file
    with open(filename, 'w') as f:
        f.writelines(output_lines)

def sample_shap_data(X, y, n_samples=2000, anomaly_ratio=0.3, random_seed=42):
    np.random.seed(random_seed)
    # Find indices for normal and each anomaly class
    normal_indices = np.where(y == 0)[0]
    anomaly_classes = np.unique(y[y != 0])  # Get unique anomaly classes
    anomaly_indices = {cls: np.where(y == cls)[0] for cls in anomaly_classes}

    # Calculate the number of samples for anomalies and normal
    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    # Divide anomaly samples equally among anomaly classes
    n_anomaly_per_class = n_anomaly // len(anomaly_classes)

    # Sample anomaly indices equally from each class
    sampled_anomaly_indices = []
    for cls, indices in anomaly_indices.items():
        sampled_indices = np.random.choice(indices, n_anomaly_per_class, replace=False)
        sampled_anomaly_indices.extend(sampled_indices)

    # Sample normal indices
    sampled_normal_indices = np.random.choice(normal_indices, n_normal, replace=False)

    # Combine and shuffle the sampled indices
    selected_indices = np.concatenate([sampled_anomaly_indices, sampled_normal_indices])
    np.random.shuffle(selected_indices)

    # Get the sampled data
    X_shap = X[selected_indices]
    y_shap = y[selected_indices]

    return X_shap, y_shap

def sample_data(X, y, n_samples=2000, anomaly_ratio=0.3, random_seed=42):
    np.random.seed(random_seed)
    
    # Find indices for normal and anomaly classes
    normal_indices = np.where(y == 0)[0]
    anomaly_classes, anomaly_counts = np.unique(y[y != 0], return_counts=True)
    print(anomaly_counts)
    
    # Total anomalies in dataset
    total_anomalies = sum(anomaly_counts)
    
    # Calculate the number of samples for anomalies and normal
    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    # Determine how many samples to take from each anomaly class, proportionally
    anomaly_ratios = anomaly_counts / total_anomalies  # Proportion of each anomaly class
    n_anomaly_per_class = (n_anomaly * anomaly_ratios).astype(int)  # Allocate samples accordingly

    # Sample anomaly indices proportionally
    sampled_anomaly_indices = []
    for cls, n_samples_cls in zip(anomaly_classes, n_anomaly_per_class):
        indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(indices, min(n_samples_cls, len(indices)), replace=False)
        sampled_anomaly_indices.extend(sampled_indices)

    # Sample normal indices
    sampled_normal_indices = np.random.choice(normal_indices, min(n_normal, len(normal_indices)), replace=False)

    # Combine and shuffle the sampled indices
    selected_indices = np.concatenate([sampled_anomaly_indices, sampled_normal_indices])
    np.random.shuffle(selected_indices)

    # Get the sampled data
    X_shap = X[selected_indices]
    y_shap = y[selected_indices]

    return X_shap, y_shap


def get_shap_explanations(model, X_train, X_shap, feature_names, filename='output/shap_summary.png'):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_shap)

    #shap.summary_plot(shap_values, X_shap, feature_names=feature_names, max_display=150, show=False)
    # Save the summary plot to a file
    #plt.savefig(filename)
    #plt.close()
    print(shap_values.shape)
    for class_idx in range(shap_values.shape[2]):
        class_shap_values = shap_values[:, :, class_idx]
        # Generate class-specific summary plot
        shap.summary_plot(class_shap_values, X_shap, feature_names=feature_names, max_display=15, show=False)
        plt.savefig(f"output/shap_plots/shap_summary_class_{class_idx}.png", bbox_inches="tight")
        plt.close()
        shap.summary_plot(class_shap_values, X_shap, feature_names=feature_names, plot_type="bar", max_display=15, show=False)
        plt.savefig(f"output/shap_plots/shap_summary_class_bar_{class_idx}.png", bbox_inches="tight")
        plt.close()


        print(f"Saved SHAP plots for class {class_idx} to output/")

def get_shap_explanations_mlp(model, X_train, X_shap, feature_names, filename='output/shap_plots/shap_importance_amaretto_NN'):
    
    model.eval()  # Put the model in evaluation mode
    def model_predict(x):
        """Wrapper for SHAP to make predictions with the MLP model."""
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_tensor)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, X_train, feature_names=feature_names)

    shap_values = explainer(X_shap)  # `shap_values` is a list for multiclass

    print(f"SHAP values shape: {np.array(shap_values.values).shape}")


    for class_idx in range(len(shap_values.values)):
        class_shap_values = shap_values[:, :, class_idx]
        
        # Beeswarm plot for the current class
        shap.plots.beeswarm(class_shap_values, max_display=14)
        plt.savefig(f"{filename}_class_{class_idx}_beeswarm.png", bbox_inches="tight")
        plt.close()

        # Bar plot for the mean SHAP values for the current class
        shap.plots.bar(class_shap_values.abs.mean(0), max_display=14)
        plt.savefig(f"{filename}_class_{class_idx}_bar_mean.png", bbox_inches="tight")
        plt.close()

        print(f"Saved SHAP plots for class {class_idx} to output/shap_plots/")

    print("SHAP explanations generated and saved.")

def save_confusion_matrix(y_true, y_pred, labels=None, filename="output/confusion_matrix_m.png"):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        colorbar=False
    )

    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


def show_fraud_cases(df: pl.DataFrame, timestamps: list) -> pl.DataFrame:
    timestamps_series = pl.Series(timestamps)
    filtered_df = df.filter((pl.col("timestamp").is_in(timestamps_series)) & (pl.col("is_fraud") == "Yes"))
    return filtered_df

def find_records_around_timestamp(df: pl.DataFrame, target_timestamp: int, customer_id: int, N: int) -> pl.DataFrame:
    customer_df = df.filter(pl.col("customer.id") == customer_id)
    target_index = customer_df.get_column("timestamp").to_list().index(target_timestamp)
    start_index = max(0, target_index - N)
    end_index = min(len(customer_df), target_index + N + 1)
    result_df = customer_df[start_index:end_index]
    return result_df


def print_results(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)[1]
    recall = recall_score(y_true, y_pred, average=None)[1]
    f1 = f1_score(y_true, y_pred, average=None)[1]

    results = (
        f"Confusion Matrix:\n{cm}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
    )
    return results

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def evaluate_performance_mm(model, X_train, X_test, y_train, y_test):
    output_lines = []

    def evaluate_predictions(X, y, label):
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y, y_pred)
        precision_weighted = precision_score(y, y_pred, average="macro")
        recall_weighted = recall_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="macro")
        
        # Calculate class-specific metrics
        class_precision = precision_score(y, y_pred, average=None)
        class_recall = recall_score(y, y_pred, average=None)
        class_f1 = f1_score(y, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
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
        
        return results

    # Evaluate on train and test data
    output_lines.append(evaluate_predictions(X_train, y_train, "Train"))
    output_lines.append(evaluate_predictions(X_test, y_test, "Test"))
    
    # Save results to a file
    with open('output/performance_evaluation_multiclass.txt', 'w') as f:
        f.writelines(output_lines)

    # Print results
    for line in output_lines:
        print(line)
