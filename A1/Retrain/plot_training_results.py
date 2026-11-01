import torch
import json
import os
import matplotlib.pyplot as plt
# ==========================================
# HÀM VẼ ĐỒ THỊ LOSS & ACCURACY
# ==========================================
def plot_training_results(history_filepath="resnet18_cifar100_history.json"):
    """
    Đọc dữ liệu từ file JSON và vẽ đồ thị so sánh Train/Val Loss và Accuracy.
    """
    if not os.path.exists(history_filepath):
        print(f"Không tìm thấy file '{history_filepath}' để vẽ đồ thị.")
        return

    with open(history_filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    # Khởi tạo giao diện đồ thị
    plt.figure(figsize=(15, 5))
    
    # Biểu đồ 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'o-', color='#ff7f0e', label='Val Loss', linewidth=2)
    
    # Đánh dấu điểm Val Loss nhỏ nhất
    min_val_loss_idx = history['val_loss'].index(min(history['val_loss']))
    plt.plot(min_val_loss_idx + 1, history['val_loss'][min_val_loss_idx], 'r*', markersize=12, label='Min Val Loss')
    
    plt.title('Loss Curve (Train vs Validation)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    # Biểu đồ 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'o-', color='#2ca02c', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'o-', color='#d62728', label='Val Accuracy', linewidth=2)
    
    # Đánh dấu điểm Val Acc cao nhất
    max_val_acc_idx = history['val_acc'].index(max(history['val_acc']))
    plt.plot(max_val_acc_idx + 1, history['val_acc'][max_val_acc_idx], 'b*', markersize=12, label=f'Best Val Acc ({max(history["val_acc"]):.2f}%)')

    plt.title('Accuracy Curve (Train vs Validation)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()