import pandas as pd
import matplotlib.pyplot as plt


import os

# 设置数据块的大小
chunk_size = 48  # 可以根据你的数据量和需求调整这个值

# 读取CSV文件，使用chunksize分块读取
filepath = './pre_results/AustraliaNew_data_comparison.csv'  # 替换为你的CSV文件路径
chunks = pd.read_csv(filepath, parse_dates=[0], chunksize=chunk_size)

# 创建一个目录来保存图表
if not os.path.exists('plots'):
    os.makedirs('plots')

# 对每个数据块进行迭代
for i, chunk in enumerate(chunks):
    # 绘制每个数据块的图表
    plt.figure(figsize=(10, 6))  # 可以调整图形大小
    plt.plot(chunk.iloc[:, 0], chunk.iloc[:, 1], label='Actual')
    plt.plot(chunk.iloc[:, 0], chunk.iloc[:, 2], label='Predicted')
    plt.legend()
    plt.title(f'Plot {i + 1}')
    plt.xlabel('Date Time')
    plt.ylabel('Value')

    # 保存图表
    plt.savefig(f'plots/plot_{i + 1}.png')
    plt.close()  # 关闭图形，以便于生成下一张图表
    if i > 15:
        break

print('All plots have been saved.')