import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison_all(X_embedded, y_pred, y_true, custom_palette, all_labels, reducer, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- First plot: Predicted Classes ---
    sns.scatterplot(ax=axes[0], x=X_embedded[y_pred == 0, 0], y=X_embedded[y_pred == 0, 1], 
                    hue=y_pred[y_pred == 0], palette=custom_palette, 
                    s=30, style=y_pred[y_pred == 0], hue_order=all_labels, 
                    style_order=all_labels, legend=False)

    sns.scatterplot(ax=axes[0], x=X_embedded[y_pred != 0, 0], y=X_embedded[y_pred != 0, 1], 
                    hue=y_pred[y_pred != 0], palette=custom_palette, 
                    s=30, style=y_pred[y_pred != 0], hue_order=all_labels, 
                    style_order=all_labels, legend="full")

    axes[0].set_title(f'{reducer} with Predicted Classes', fontsize=14)
    axes[0].set_xlabel(f'{reducer} Component 1', fontsize=12)
    axes[0].set_ylabel(f'{reducer} Component 2', fontsize=12)

    # --- Second plot: True Classes ---
    sns.scatterplot(ax=axes[1], x=X_embedded[y_true == 0, 0], y=X_embedded[y_true == 0, 1], 
                    hue=y_true[y_true == 0], palette=custom_palette, 
                    s=30, style=y_true[y_true == 0], hue_order=all_labels, 
                    style_order=all_labels, legend=False)

    sns.scatterplot(ax=axes[1], x=X_embedded[y_true != 0, 0], y=X_embedded[y_true != 0, 1], 
                    hue=y_true[y_true != 0], palette=custom_palette, 
                    s=30, style=y_true[y_true != 0], hue_order=all_labels, 
                    style_order=all_labels, legend="full")

    axes[1].set_title(f'{reducer} with True Classes', fontsize=14)
    axes[1].set_xlabel(f'{reducer} Component 1', fontsize=12)
    axes[1].set_ylabel(f'{reducer} Component 2', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_comparison_anomalies(X_embedded, y_pred, y_true, custom_palette, all_labels, reducer, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

    # --- First plot: Predicted Classes ---
    sns.scatterplot(ax=axes[0], x=X_embedded[:, 0], y=X_embedded[:, 1], 
                    hue=y_pred, palette=custom_palette, 
                    s=30, style=y_pred, hue_order=all_labels, 
                    style_order=all_labels, legend=False)

    axes[0].set_title(f'{reducer} with Predicted Anomaly Types', fontsize=14)
    axes[0].set_xlabel(f'{reducer} Component 1', fontsize=12)
    axes[0].set_ylabel(f'{reducer} Component 2', fontsize=12)

    # --- Second plot: True Classes ---
    sns.scatterplot(ax=axes[1], x=X_embedded[y_true == 0, 0], y=X_embedded[y_true == 0, 1], 
                    hue=y_true[y_true == 0], palette=custom_palette, 
                    s=30, style=y_true[y_true == 0], hue_order=all_labels, 
                    style_order=all_labels, legend=False)

    sns.scatterplot(ax=axes[1], x=X_embedded[y_true != 0, 0], y=X_embedded[y_true != 0, 1], 
                    hue=y_true[y_true != 0], palette=custom_palette, 
                    s=30, style=y_true[y_true != 0], hue_order=all_labels, 
                    style_order=all_labels, legend="full")

    axes[1].set_title(f'{reducer} with True Anomaly Types', fontsize=14)
    axes[1].set_xlabel(f'{reducer} Component 1', fontsize=12)
    axes[1].set_ylabel(f'{reducer} Component 2', fontsize=12)

    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
