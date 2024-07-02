import numpy as np
import matplotlib.pyplot as plt
import os

# 示例文件路径
folder_path = './results/loss_flag1_lr0.0001_dm256_test_DLinear_Load_ftMS_sl96_pl24_p16s8_random2021_0'  # 修改为你的实验名称

# 加载数据
preds = np.load(os.path.join(folder_path, 'pred.npy'))
trues = np.load(os.path.join(folder_path, 'true.npy'))

# 计算平均值用于绘图
mean_preds = np.mean(preds, axis=0)
mean_trues = np.mean(trues, axis=0)

# 设置图表风格
plt.style.use('ggplot')

# 设置字体大小
plt.rcParams.update({'font.size': 14})

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制预测值和真实值
plt.plot(mean_trues, label='True Values', color='blue', linewidth=2)
plt.plot(mean_preds, label='Predicted Values', color='red', linestyle='--', linewidth=2)

# 添加图例和标签
plt.xlabel('Time Steps', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.title('True vs Predicted Values', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)

# 添加sci论文风格的注释
plt.tight_layout()

# 保存为PDF
plt.savefig(os.path.join(folder_path, 'comparison_plot.pdf'), format='pdf')

# 显示图表
plt.show()
