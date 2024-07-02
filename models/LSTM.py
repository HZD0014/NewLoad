from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.input_dim  # 假设在configs中有定义
        self.hidden_dim = configs.hidden_dim  # 假设在configs中有定义
        self.num_layers = configs.num_layers  # 假设在configs中有定义

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.Linear = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()  # 保持最后一个时间步的值

        # LSTM expects input of shape (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: [Batch, Seq_len, Hidden_dim]

        # Select the output of the last time step from the LSTM
        lstm_out_last = lstm_out[:, -1, :]  # lstm_out_last: [Batch, Hidden_dim]

        # Linear layer expects input of shape (batch_size, hidden_dim)
        pred = self.Linear(lstm_out_last)  # pred: [Batch, Pred_len]

        # Broadcast seq_last to match the shape of pred
        seq_last_expanded = seq_last.expand(-1, self.pred_len, -1)

        # Add the last sequence value back to the prediction
        pred = pred.unsqueeze(-1) + seq_last_expanded  # pred: [Batch, Pred_len, Channel]

        return pred