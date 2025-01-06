import matplotlib.pyplot as plt
import numpy as np

def loss_and_acc(loss_train,loss_val,acc_train,acc_val,out_path):

    plt.figure(figsize=(8, 6))  # 可以设置图表的大小
    plt.plot(np.array(loss_train), label='Train Loss', color='blue')
    plt.plot(np.array(loss_val), label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper left')

    ax2 = plt.gca().twinx()
    ax2.plot(np.array(acc_train), label='Train Accuracy', color='green', linestyle='--')
    ax2.plot(np.array(acc_val), label='Validation Accuracy', color='red', linestyle='--')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.legend(loc='upper right')

    plt.savefig(out_path+'loss_acc.pdf')