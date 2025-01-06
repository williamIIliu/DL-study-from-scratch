from sklearn.metrics import roc_auc_score, f1_score, hamming_loss,auc,roc_curve
import matplotlib.pyplot as plt
import numpy as np


def performance(y_true,y_pred,result_path):
    results = []
    # ROC AUC
    try:
        auc_score = roc_auc_score(y_true, y_pred, average='macro')
        results.append(f"Test ROC-AUC: {auc_score:.4f}")
    except ValueError as e:
        results.append(f"ROC AUC Error: {e}")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # 计算 AUC 值
    roc_auc = auc(fpr, tpr)
    # results.append(f"ROC AUC: {roc_auc:.4f}")

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(result_path+'ROC.png')

    # F1 Score
    y_pred_binary = (y_pred > 0.5).astype(int)
    try:
        f1 = f1_score(y_true, y_pred_binary, average='macro')
        results.append(f"Test F1 Score: {f1:.4f}")
    except ValueError as e:
        results.append(f"F1 Score Error: {e}")

    # Hamming Loss
    try:
        hamming = hamming_loss(y_true, y_pred_binary)
        results.append(f"Test Hamming Loss: {hamming:.4f}")
    except ValueError as e:
        results.append(f"Hamming Loss Error: {e}")

    return results

if __name__ == '__main__':
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=100)  # 真实标签
    y_pred = np.random.rand(100)  # 预测概率
    folder = '../logs/'
    results = performance(y_true,y_pred,folder)
    print(results)