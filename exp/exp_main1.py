from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, Linear, NLinear, PatchMixer, FITS, FreTS, MPLinear, MPFreTS, LSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, MAE, MSE, MAPE, RMSE
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import matplotlib.pyplot as plt
import torchvision

warnings.filterwarnings('ignore')


# 自定义多任务损失函数，结合L1和L2损失
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        l1_loss = self.l1_loss(outputs, targets)
        l2_loss = self.l2_loss(outputs, targets)
        return self.alpha * l1_loss + self.beta * l2_loss


# 主实验类，继承自Exp_Basic
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    # 构建模型
    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchMixer': PatchMixer,
            'FITS': FITS,
            'FreTS': FreTS,
            'MPLinear': MPLinear,
            'MPFreTS': MPFreTS,
            'LSTM' : LSTM
        }
        model = model_dict[self.args.model].Model(self.args).float().to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据集和数据加载器
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器
    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    # 选择损失函数
    def _select_criterion(self):
        if self.args.loss_flag == 1:
            return nn.L1Loss()
        elif self.args.loss_flag == 2:
            return MultiTaskLoss(alpha=0.5, beta=0.5)
        elif self.args.loss_flag == 3:
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()

    # 计算指标


    # 验证模型
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, date_stamp in vali_loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if f_dim == -1:
                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].to(self.device, non_blocking=True)

                pred, true = outputs.detach().cpu(), batch_y.detach().cpu()
                loss = criterion(pred, true)
                preds.append(pred.numpy())
                trues.append(true.numpy())
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        mae, rmse, r2 = metric(preds, trues)
        self.model.train()
        return total_loss, mae, rmse, r2

    # 训练模型
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=len(train_loader),
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        train_loss = []
        for epoch in range(self.args.train_epochs):  # 修改epoch数目以适应实验
            epoch_loss = []
            preds, trues = [], []
            self.model.train()
            epoch_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark, date_stamp in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if f_dim == -1:
                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].to(self.device, non_blocking=True)

                loss = criterion(outputs, batch_y)
                epoch_loss.append(loss.item())
                loss.backward()
                model_optim.step()


                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
            scheduler.step()
            print("Epoch: {} 耗时: {}".format(epoch + 1, time.time() - epoch_time))
            epoch_loss = np.average(epoch_loss)
            preds, trues = np.concatenate(preds), np.concatenate(trues)
            train_mae, train_rmse, train_r2 = metric(preds, trues)
            vali_loss, vali_mae, vali_rmse, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch: {epoch + 1}, | 训练损失: {epoch_loss:.7f}, 验证损失: {vali_loss:.7f}")
            print(f"训练集: MAE: {train_mae:.7f}, RMSE: {train_rmse:.7f}, R2: {train_r2:.7f}%")
            print(f"验证集: MAE: {vali_mae:.7f}, RMSE: {vali_rmse:.7f}, R2: {vali_r2:.7f}%")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("早停")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    # 测试模型
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test == 1:
            print('加载模型')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')))

        preds, trues, date_stamp_datetimes = [], [], []

        losss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, date_stamp) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if f_dim == -1:
                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].to(self.device, non_blocking=True)
                # 初始化一个空列表来存储转换后的datetime对象


                # 使用for循环遍历date_stamp的每一行
                for row in date_stamp:
                    # 将每一行转换为datetime类型，并添加到列表中
                    t = pd.to_datetime(row, unit='s')
                    date_stamp_datetimes.append(t[0])


                pred, true = outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()
                losss.append(metric(pred, true))
                preds.extend(pred[:, 0])
                trues.extend(true[:, 0])




        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds, trues = np.array(preds), np.array(trues)
        # 假设date_stamp_datetimes, trues, preds是已经定义好的列表或数组
        df = pd.DataFrame({
            'dates': date_stamp_datetimes,
            'trues': trues,
            # 使用f-string创建新列名
            f'{self.args.model}_preds': preds,
        })

        # 确定文件名和路径
        file_name = f"./pre_results/{self.args.data_path.split('.')[0]}_data_comparison.csv"

        # 检查文件是否存在
        if os.path.isfile(file_name):
            # 如果文件存在，读取CSV文件
            existing_df = pd.read_csv(file_name)
            # 直接用新数据覆盖现有列
            existing_df[f'{self.args.model}_preds'] = df[f'{self.args.model}_preds']
        else:
            # 如果文件不存在，使用初始DataFrame
            existing_df = df

        # 保存DataFrame到CSV文件，不包含索引
        existing_df.to_csv(file_name, index=False)

        folder_path = os.path.join('./metric_results', setting)
        os.makedirs(folder_path, exist_ok=True)
        # 将列表转换为NumPy数组以方便计算
        loss_array = np.array(losss)

        # 使用np.mean沿着数组的第一个维度计算每个损失指标的平均值
        mae, rmse, r2 = np.mean(loss_array, axis=0)
        # 确保所有指标都是标量
        mae = np.mean(mae) if isinstance(mae, (np.ndarray, list)) else mae
        rmse = np.mean(rmse) if isinstance(rmse, (np.ndarray, list)) else rmse
        r2 = np.mean(r2) if isinstance(r2, (np.ndarray, list)) else r2

        # 格式化输出测试结果
        print("Test Performance Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae}")
        print(f"  Mean Squared Error (RMSE): {rmse}")

        print(f"   Mean Absolute Percentage Error (R2): {r2}")

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, rmse, r2]))



    # 预测
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x) if 'Linear' in self.args.model else self.model(batch_x, batch_x_mark)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
