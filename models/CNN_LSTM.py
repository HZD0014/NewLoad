from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, configs):
        super(CNNLSTM, self).__init__()
        # 从配置中获取模型参数
        self.input_channel = configs.input_channel  # 输入通道数
        self.output_channel = configs.output_channel  # 输出通道数
        self.kernel_size = configs.kernel_size  # 卷积核大小
        self.lstm_units = configs.lstm_units  # LSTM单元数
        self.output_length = configs.output_length  # 输出序列长度
        self.cnn_layers = configs.cnn_layers  # CNN层数
        self.num_layers = configs.num_layers  # LSTM层数

        # 定义CNN层，使用多个卷积层来提取特征
        self.cnn = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=self.input_channel if i == 0 else self.output_channel,  # 输入通道
                    out_channels=self.output_channel,  # 输出通道
                    kernel_size=self.kernel_size,  # 卷积核大小
                    padding=(self.kernel_size - 1) // 2  # 填充以保持输出尺寸不变
                )
                for i in range(self.cnn_layers)  # 循环创建多层卷积层
            ]
        )

        # 定义LSTM层，用于处理序列数据
        self.lstm = nn.LSTM(
            input_size=self.output_channel,  # 输入特征维度
            hidden_size=self.lstm_units,  # 隐藏层维度
            num_layers=self.num_layers,  # LSTM堆叠层数
            batch_first=True  # 输入和输出张量的第一个维度为批次大小
        )

        # 定义线性层，将LSTM的输出映射到最终的输出维度
        self.linear = nn.Linear(self.lstm_units, self.output_length * self.output_channel)

    def forward(self, x):
        # x: [Batch, Input length, Channel] 输入数据
        # 转换数据维度以适应CNN层
        x = x.permute(0, 2, 1)  # 转换为 [Batch, Channel, Input length]

        # 通过CNN层提取特征
        x = self.cnn(x)

        # 转换回原始数据维度以适应LSTM层
        x = x.permute(0, 2, 1)  # 转换回 [Batch, Input length, Channel]

        # 将数据重新排列以满足LSTM的输入要求
        x = x.contiguous().view(x.size(0), -1, self.output_channel)  # [Batch, Sequence length, Features]

        # 通过LSTM层处理序列数据
        lstm_out, _ = self.lstm(x)

        # 取LSTM最后一个时间步的输出用于预测
        last_time_step = lstm_out[:, -1, :]  # [Batch, LSTM units]

        # 通过线性层将LSTM输出映射到最终输出维度
        out = self.linear(last_time_step)  # [Batch, Output length * Channel]
        # 重新排列输出维度以匹配预期的输出格式
        out = out.view(-1, self.output_length, self.output_channel)  # [Batch, Output length, Channel]

        return out  # 返回模型预测结果