import argparse
import os
import types

import torch
from exp.exp_main1 import Exp_Main
import random
import numpy as np

# 定义不同数据集和模型参数
datasets = [
    {
        'dataset': 'Load',
        'root_path': './dataset/',
        'data_path': 'AustraliaNew.csv',
        'features': 'MS',
        'target': 'LOAD',
        'freq': 'h',
        'enc_in': 33,
        'seq_len': 192,
        'label_len': 0,
        'pred_len': 48,
    },
    {
        'dataset': 'Load',
        'root_path': './dataset/',
        'data_path': 'PanamaLoad1.csv',
        'features': 'MS',
        'target': 'LOAD',
        'freq': 'h',
        'enc_in': 37,
        'seq_len': 96,
        'label_len': 0,
        'pred_len': 24,
    }
]

model_params = {
    'patch_len': 16,
    'stride': 8,
    'padding_patch': 'end',
    'revin': 1,
    'affine': 0,
    'subtract_last': 0,
    'decomposition': 0,
    'kernel_size': 25,
    'individual': 0,
    'embed_type': 0,
    'mixer_kernel_size': 8,
    'd_model': 256,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'moving_avg': 25,
    'factor': 1,
    'use_dct': True,
    'dropout1': 0.2,
    'patch_list': [12, 24, 48],
    'top_k': 3,
    'conv_kernel': [3, 5, 7],
}

params = {
    'random_seed': 2021,
    'is_training': 1,
    'model_id': 'test',
    'model': 'MPFreTS',
    'des': 'test',
    'checkpoints': './checkpoints/',
    'fc_dropout': 0.05,
    'head_dropout': 0.,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'do_predict': False,
    'num_workers': 6,
    'itr': 1,
    'train_epochs': 300,
    'batch_size': 128,
    'patience': 30,
    'learning_rate': 0.00005,
    'lradj': 'constant',
    'pct_start': 0.3,
    'use_amp': False,
    'use_gpu': True,
    'gpu': 0,
    'use_multi_gpu': False,
    'devices': '0',
    'test_flop': False,
    'loss_flag': 1,
}

def main():
    # 设定随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    for dataset in datasets:
        args = types.SimpleNamespace(**(params | model_params | dataset))
        # 检查是否使用 GPU
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        print('Args in experiment:')
        print(args)

        Exp = Exp_Main

        if args.is_training:
            for ii in range(args.itr):
                # 设置实验记录
                setting = (
                    f'loss_flag{args.loss_flag}_'
                    f'lr{args.learning_rate}_'
                    f'dm{args.d_model}_'
                    f'data-{args.data_path.split(".")[0]}_'
                    f'model{args.model}_'
                    f'{args.dataset}_'
                    f'sl{args.seq_len}_'
                    f'pl{args.pred_len}_'
                    f'p{args.patch_len}_'
                    f's{args.stride}_'
                    f'random{args.random_seed}_'
                    f'{ii}'
                )

                exp = Exp(args)  # 设置实验
                print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                exp.train(setting)

                print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.test(setting, test=0)

                if args.do_predict:
                    print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exp.predict(setting, True)

                torch.cuda.empty_cache()

        else:
            ii = 0
            setting = (
                f'loss_flag{args.loss_flag}_'
                f'lr{args.learning_rate}_'
                f'dm{args.d_model}_'
                f'data-{args.data_path.split(".")[0]}_'
                f'model{args.model}_'
                f'{args.dataset}_'
                f'sl{args.seq_len}_'
                f'pl{args.pred_len}_'
                f'p{args.patch_len}_'
                f's{args.stride}_'
                f'random{args.random_seed}_'
                f'{ii}'
            )

            exp = Exp(args)  # 设置实验
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
