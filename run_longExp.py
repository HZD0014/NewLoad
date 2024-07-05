import argparse
import os
import types

import torch
from exp.exp_main1 import Exp_Main
import random
import numpy as np

params = {
    # 基础配置
    'random_seed': 2021,  # 随机种子，确保实验的可重复性
    'is_training': 1,  # 训练状态，1 表示训练模式，0 表示测试模式
    'model_id': 'test',  # 模型的唯一标识符
    'model': 'MPLinear',  # 模型名称
    'des': 'test',  # 实验描述

    # 数据配置
    'dataset': 'Load',  # 数据集类型
    'root_path': './dataset/',  # 数据集文件的根路径
    'data_path': 'AustraliaNew.csv',  # 数据集文件名
    'features': 'MS',  # 预测任务类型
    'target': 'LOAD',  # 目标特征
    'freq': 'h',  # 时间特征编码的频率

    # 模型结构参数
    'seq_len': 96,  # 输入序列长度
    'label_len': 0,  # 起始标记长度
    'pred_len': 24,  # 预测序列长度
    'patch_len': 16,  # 补丁长度
    'stride': 8,  # 步幅
    'padding_patch': 'end',  # 补丁的填充方式
    'revin': 1,  # 是否使用 RevIN
    'affine': 0,  # 是否使用 RevIN 的仿射变换
    'subtract_last': 0,  # RevIN 的减法策略
    'decomposition': 0,  # 是否使用分解
    'kernel_size': 25,  # 分解的卷积核大小
    'individual': 0,  # 是否使用单独的头
    'embed_type': 0,  # 嵌入类型

    # 模型超参数
    'enc_in': 33,  # 特征数量   Australia : 33
    'mixer_kernel_size': 8,  # PatchMixer 的卷积核大小
    'd_model': 256,  # 模型维度
    'n_heads': 8,  # 注意力头的数量
    'e_layers': 2,  # 编码器层数
    'd_layers': 1,  # 解码器层数
    'd_ff': 2048,  # 前馈神经网络的维度
    'moving_avg': 25,  # 移动平均的窗口大小
    'factor': 1,  # 注意力因子

    # 训练配置
    'checkpoints': './checkpoints/',  # 模型检查点保存位置
    'fc_dropout': 0.05,  # 全连接层的 dropout 率
    'head_dropout': 0.,  # 注意力头的 dropout 率
    'dropout': 0.1,  # dropout 率
    'embed': 'timeF',  # 时间特征编码类型
    'activation': 'gelu',  # 激活函数
    'output_attention': False,  # 是否在编码器中输出注意力
    'do_predict': False,  # 是否预测未来数据
    'num_workers': 6,  # 数据加载器的工作线程数
    'itr': 2,  # 实验次数
    'train_epochs': 300,  # 训练周期数
    'batch_size': 128,  # 训练输入数据的批量大小
    'patience': 30,  # 早停法的耐心值

    # 优化器配置
    'learning_rate': 0.00005,  # 优化器的学习率
    'lradj': 'constant',  # 学习率调整策略
    'pct_start': 0.3,  # pct_start 参数

    # 硬件配置
    'use_amp': False,  # 是否使用自动混合精度训练
    'use_gpu': True,  # 是否使用 GPU
    'gpu': 0,  # 使用的 GPU 编号
    'use_multi_gpu': False,  # 是否使用多 GPU
    'devices': '0',  # 多 GPU 的设备 ID

    # 其他配置
    'test_flop': False,  # 是否测试 FLOP
    'loss_flag': 1 , # 损失函数标志

    'dropout1': 0.2,
    'patch_list': [12, 24, 48],
    'top_k': 3,
    'conv_kernel': [3, 5, 7],
}


def main():
    args = types.SimpleNamespace(**params)
    # 设定随机种子
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 检查是否使用 GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # 设置实验记录
            setting = 'loss_flag{}_lr{}_dm{}_{}_data-{}_model{}_{}_ft{}_sl{}_pl{}_p{}_s{}_random{}_{}'.format(
                args.loss_flag,
                args.learning_rate,
                args.d_model,
                args.model_id,
                args.data_path.split('.')[0],
                args.model,
                args.dataset,
                args.features,
                args.seq_len,
                args.pred_len,
                args.patch_len,
                args.stride,
                args.random_seed, ii)

            exp = Exp(args)  # 设置实验
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=0)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = 'loss_flag{}_lr{}_dm{}_{}_{}_{}_ft{}_sl{}_pl{}_p{}s{}_random{}_{}'.format(
            args.loss_flag,
            args.learning_rate,
            args.d_model,
            args.model_id,
            args.model,
            args.dataset,
            args.features,
            args.seq_len,
            args.pred_len,
            args.patch_len,
            args.stride,
            args.random_seed, ii)

        exp = Exp(args)  # 设置实验
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
