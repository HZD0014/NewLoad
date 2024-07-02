import types

from torchsummary import summary

from models import DLinear, Linear, NLinear, PatchMixer, SegRNN, DCT_LSTM,FITS
import torch
params = {
    'random_seed': 2021,  # 随机种子，用于确保实验的可重复性
    'is_training': 1,  # 训练状态，1 表示训练模式，0 表示测试模式
    'model_id': 'test',  # 模型的唯一标识符
    'model': 'DLinear',  # 模型名称，可选值: [Autoformer, Informer, Transformer]
    'dataset': 'Load',  # 数据集类型
    'root_path': './dataset/',  # 数据集文件的根路径
    'data_path': 'PanamaLoad.csv',  # 数据集文件名
    'features': 'MS',  # 预测任务类型，M: 多变量预测多变量，S: 单变量预测单变量，MS: 多变量预测单变量
    'target': 'LOAD',  # 在 S 或 MS 任务中指定的目标特征
    'freq': 'h',  # 时间特征编码的频率
    'checkpoints': './checkpoints/',  # 模型检查点保存的位置
    'seq_len': 96,  # 输入序列的长度
    'label_len': 0,  # 起始标记的长度
    'pred_len': 24,  # 预测序列的长度
    'fc_dropout': 0.05,  # 全连接层的 dropout 率
    'head_dropout': 0.0,  # 注意力头的 dropout 率
    'patch_len': 16,  # 补丁长度
    'stride': 8,  # 步幅
    'padding_patch': 'end',  # 补丁的填充方式，None: 无填充；end: 在末尾填充
    'revin': 1,  # 是否使用 RevIN，1 表示使用，0 表示不使用
    'affine': 0,  # 是否使用 RevIN 的仿射变换，1 表示使用，0 表示不使用
    'subtract_last': 0,  # RevIN 的减法策略，0 表示减去均值，1 表示减去最后一个值
    'decomposition': 0,  # 是否使用分解，1 表示使用，0 表示不使用
    'kernel_size': 25,  # 分解的卷积核大小
    'individual': 0,  # 是否使用单独的头，1 表示使用，0 表示不使用
    'embed_type': 0,  # 嵌入类型，0: 默认，1: 值嵌入 + 时间嵌入 + 位置嵌入，2: 值嵌入 + 时间嵌入，3: 值嵌入 + 位置嵌入，4: 仅值嵌入

    'enc_in': 36,  # 特征数量，Australia：32，Panama：36

    'mixer_kernel_size': 8,  # PatchMixer 的卷积核大小
    'd_model': 256,  # 模型维度
    'n_heads': 8,  # 注意力头的数量
    'e_layers': 2,  # 编码器层数
    'd_layers': 1,  # 解码器层数
    'd_ff': 2048,  # 前馈神经网络的维度
    'moving_avg': 25,  # 移动平均的窗口大小
    'factor': 1,  # 注意力因子

    'dropout': 0.05,  # dropout 率
    'embed': 'timeF',  # 时间特征编码类型，选项: [timeF, fixed, learned]
    'activation': 'gelu',  # 激活函数
    'output_attention': False,  # 是否在编码器中输出注意力
    'do_predict': False,  # 是否预测未来数据
    'num_workers': 10,  # 数据加载器的工作线程数
    'itr': 2,  # 实验次数
    'train_epochs': 150,  # 训练周期数
    'batch_size': 128,  # 训练输入数据的批量大小
    'patience': 100,  # 早停法的耐心值
    'learning_rate': 0.0001,  # 优化器的学习率
    'des': 'test',  # 实验描述

    'lradj': 'type3',  # 学习率调整策略
    'pct_start': 0.3,  # pct_start 参数
    'use_amp': True,  # 是否使用自动混合精度训练
    'use_gpu': True,  # 是否使用 GPU
    'gpu': 0,  # 使用的 GPU 编号
    'use_multi_gpu': False,  # 是否使用多 GPU
    'devices': '0,1',  # 多 GPU 的设备 ID
    'test_flop': False,  # 是否测试 FLOP

    'rnn_type' : 'lstm',
    'dec_way' : 'rmf',
    'loss_flag': 1 , # 损失函数标志，0 表示 MSE，1 表示 MAE，2 表示同时使用 MSE 和 MAE，3 表示 SmoothL1Loss
    'cut_freq' : 0
}

# 将字典转换为命名空间
configs = types.SimpleNamespace(**params)
models_dict = {
    'DLinear' : DLinear,
    'PatchMixer': PatchMixer,

    'FITS': FITS,
}
for modelName, modelClass in models_dict.items():
    model = modelClass.Model(configs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"------------>>>{modelName}<<<---------------")
    summary(model, input_size=(96, 36))
    print("---------------------------------------------")
