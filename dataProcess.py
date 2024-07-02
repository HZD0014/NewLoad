import pandas as pd

# 加载CSV文件
df = pd.read_csv('./dataset/AustraliaRaw.csv')

# 转换日期列为datetime类型，并与HOUR列合并
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# 确保HOUR列是整数类型，然后乘以60（因为是半天）
df['hour'] = df['hour'].astype(float) * 60

# 将日期和小时组合成一个新的datetime列
df['date'] = df['date'] + pd.to_timedelta(df['hour'], unit='m')
# 确保datetime列是datetime类型
df['date'] = pd.to_datetime(df['date'])

# 保存到新的CSV文件中，或者覆盖源文件
df.to_csv('./dataset/AustraliaNew.csv', index=False)
