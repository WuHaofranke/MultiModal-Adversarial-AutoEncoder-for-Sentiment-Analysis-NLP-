import matplotlib.pyplot as plt
import numpy as np


def plot_results(val_loss, val_acc, test_loss, test_acc, epoch):
    epochs = range(1, epoch + 1)

    plt.figure(figsize=(12, 6))

    # 绘制验证集和测试集的损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.plot(epochs, test_loss, 'b', label='Test loss')
    plt.title('Validation and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证集和测试集的精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.plot(epochs, test_acc, 'b', label='Test accuracy')
    plt.title('Validation and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


