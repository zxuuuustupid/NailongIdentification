from matplotlib import pyplot as plt


def process_show(strm,epoch_list, loss_list, train_acclist, test_acclist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(strm, fontsize=16)
    # 左侧绘制loss
    ax1.plot(epoch_list, loss_list, color='blue')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid()

    # 右侧绘制acc
    ax2.plot(epoch_list, train_acclist, label='Training Accuracy', color='green')
    ax2.plot(epoch_list, test_acclist, label='Testing Accuracy', color='orange')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()

    # 显示图形
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('./results/'+strm+'_plot.png')
    plt.show()
