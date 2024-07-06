import pandas as pd

# 加载CSV文件
df = pd.read_csv('./dataset/PanamaLoad.csv')


# 假设第一列的列名是'date_time'
date_time_column = 'date'

# 转换时间格式，确保时间列是datetime类型
df[date_time_column] = pd.to_datetime(df[date_time_column], format='%Y/%m/%d %H:%M')

# 计算小时数并转换为分钟
df['hour'] = df[date_time_column].dt.hour * 60 + df[date_time_column].dt.minute

# 显示结果
print(df)

# 将结果保存到新的CSV文件
df.to_csv('./dataset/PanamaLoad1.csv', index=False)
