@echo off
rem 创建logs目录
if not exist .\logs mkdir .\logs
if not exist .\logs\LongForecasting mkdir .\logs\LongForecasting

set seq_len=336
set model_name=PatchMixer
set root_path_name=.\dataset\ETT-small\
set data_path_name=ETTh1.csv
set model_id_name=ETTh1
set data_name=ETTh1
set random_seed=2021

for %%i in (96 192 336 720) do (
    set pred_len=%%i
    python -u run_longExp.py ^
      --random_seed %random_seed% ^
      --is_training 1 ^
      --root_path %root_path_name% ^
      --data_path %data_path_name% ^
      --model_id %model_id_name%_sl%seq_len%_pl%pred_len% ^
      --model %model_name% ^
      --data %data_name% ^
      --features M ^
      --seq_len %seq_len% ^
      --pred_len %pred_len% ^
      --enc_in 7 ^
      --e_layers 1 ^
      --d_model 256 ^
      --dropout 0.2 ^
      --head_dropout 0 ^
      --patch_len 16 ^
      --stride 8 ^
      --des 'Exp' ^
      --train_epochs 100 ^
      --patience 10 ^
      --loss_flag 2 ^
      --itr 1 --batch_size 1024 --learning_rate 0.0001 ^
      >.\logs\LongForecasting\%model_name%_%model_id_name%_sl%seq_len%_pl%pred_len%_random_seed%random_seed%.log
)