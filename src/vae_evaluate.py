import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score, precision_score,
    recall_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
import mlflow
from vae import CNNVAE, LATENT_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLITS_DIR = './data/splits'
MODELS_DIR = './models'
OUTPUT_DIR = './outputs'


def load_model():
    model = CNNVAE(latent_dim=LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(
        f'{MODELS_DIR}/cnnvae_model.pt',
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded — best loss: {checkpoint['best_loss']:.4f} at epoch {checkpoint['epoch']}")
    
    return model


def load_test_data():

    test_X = np.load(f'{SPLITS_DIR}/test_beats.npy')
    test_y = np.load(f'{SPLITS_DIR}/test_labels.npy')
    print(f"Test beats: {test_X.shape}")
    print(f"Test labels: {test_y.shape}")
    print(f"Normal: {(test_y==0).sum():,} ({(test_y==0).mean()*100:.1f}%)")
    print(f"Anomaly: {(test_y==1).sum():,} ({(test_y==1).mean()*100:.1f}%)")
    
    return test_X, test_y


def get_reconstruction_errors(model, test_X, batch_size=512):
    
    errors = []
    model.eval()
    for i in range(0, len(test_X), batch_size):
        batch = torch.FloatTensor(test_X[i:i+batch_size]).to(DEVICE)
        error = model.reconstruction_error(batch)
        errors.append(error.cpu().numpy())
    errors = np.concatenate(errors)
    
    return errors


def evaluate():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()
    test_X, test_y = load_test_data()

    print("\nComputing reconstruction errors...")
    errors = get_reconstruction_errors(model, test_X)
    errors = (errors - errors.min()) / (errors.max() - errors.min())
    thresholds = np.percentile(errors, np.arange(50, 92, 0.5))
    best_f2, best_threshold = 0, 0

    for t in thresholds:
        preds = (errors >= t).astype(int)
        f2 = fbeta_score(test_y, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2        = f2
            best_threshold = t

    predictions = (errors >= best_threshold).astype(int)

    auroc = roc_auc_score(test_y, errors)
    auprc = average_precision_score(test_y, errors)
    f1 = f1_score(test_y, predictions)
    f2 = fbeta_score(test_y, predictions, beta=2)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)
    
    with mlflow.start_run(run_name="cnnvae_eval", nested=True):
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("f2", f2)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_artifact(f'{OUTPUT_DIR}/vae_evaluation.png')

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    plot_results(errors, test_y, best_threshold, auroc, auprc, f2, precision, recall, cm)


def plot_results(errors, labels, threshold, auroc, auprc, f2, precision, recall, cm):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle('CNN-VAE ECG Anomaly Detection — Evaluation Results', color='white', fontsize=13)
    colors = {'normal': '#00d4aa', 'anomaly': '#ff4d6d', 'line': '#f4a261'}

    ax = axes[0][0]
    ax.set_facecolor('#1a1a2e')
    normal_errors  = errors[labels == 0]
    anomaly_errors = errors[labels == 1]
    ax.hist(normal_errors,  bins=80, alpha=0.7, color=colors['normal'],  label='Normal',  density=True)
    ax.hist(anomaly_errors, bins=80, alpha=0.7, color=colors['anomaly'], label='Anomaly', density=True)
    ax.axvline(threshold, color=colors['line'], linestyle='--', linewidth=1.5, label=f'Threshold: {threshold:.3f}')
    ax.set_title('Anomaly Score Distribution', color='white')
    ax.set_xlabel('Reconstruction Error', color='gray')
    ax.set_ylabel('Density', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[0][1]
    ax.set_facecolor('#1a1a2e')
    fpr, tpr, _ = roc_curve(labels, errors)
    ax.plot(fpr, tpr, color=colors['normal'], linewidth=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0,1], [0,1], color='gray', linestyle='--', linewidth=1)
    ax.set_title('ROC Curve', color='white')
    ax.set_xlabel('False Positive Rate', color='gray')
    ax.set_ylabel('True Positive Rate', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[1][0]
    ax.set_facecolor('#1a1a2e')
    prec, rec, _ = precision_recall_curve(labels, errors)
    ax.plot(rec, prec, color=colors['anomaly'], linewidth=2, label=f'AUPRC = {auprc:.4f}')
    ax.set_title('Precision-Recall Curve', color='white')
    ax.set_xlabel('Recall', color='gray')
    ax.set_ylabel('Precision', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[1][1]
    ax.set_facecolor('#1a1a2e')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix', color='white')
    ax.set_xlabel('Predicted', color='gray')
    ax.set_ylabel('Actual', color='gray')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Anomaly'], color='gray')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Anomaly'], color='gray')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white', fontsize=14, fontweight='bold')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    metrics_text = (
        f"F2 Score: {f2:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Threshold: {threshold:.4f}"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', color='white',
            fontsize=10, family='monospace',
            bbox=dict(facecolor='#1a1a2e', edgecolor='#333', boxstyle='round'))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/vae_evaluation.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f0f')
    print(f"\nPlot saved to {OUTPUT_DIR}/vae_evaluation.png")


if __name__ == "__main__":
    
    evaluate()