import re
import matplotlib.pyplot as plt

# 从日志文件中读取数据
def read_log_file(file_path):
    top_1_list = []
    top_5_list = []
    loss_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式提取 top 1, top 5 和 loss 的值
            match = re.search(r'top_1: : ([\d.]+) %.*top_5: : ([\d.]+) %.*test loss: ([\d.]+)', line)
            if match:
                top_1 = float(match.group(1))
                top_5 = float(match.group(2))
                loss = float(match.group(3))
                
                top_1_list.append(top_1)
                top_5_list.append(top_5)
                loss_list.append(loss)
                
    return top_1_list, top_5_list, loss_list

# 绘制线型图
def plot_metrics(top_1_list, top_5_list, loss_list):
    epochs = range(1, len(top_1_list) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 绘制 Top 1 准确度
    plt.subplot(1, 3, 1)
    plt.plot(epochs, top_1_list, marker='o')
    plt.title('Top 1 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    # 绘制 Top 5 准确度
    plt.subplot(1, 3, 2)
    plt.plot(epochs, top_5_list, marker='o')
    plt.title('Top 5 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    # 绘制 Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, loss_list, marker='o')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 读取日志文件（请替换为您的日志文件路径）
    file_path = '/home/yzha9041/Oort/training/evals/logs/openimage/0830_172019/aggregator/log'
    top_1_list, top_5_list, loss_list = read_log_file(file_path)
    
    # 绘制线型图
    plot_metrics(top_1_list, top_5_list, loss_list)