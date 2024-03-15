# B2
# python train.py task=DLARBittle_PRD_v2 train=DLARBittlePPO_LSTM checkpoint=runs/DLARBittle_B2_0.1-0.8/nn/DLARBittle.pth test=True num_envs=1 torch_deterministic=True

# GP
# python train.py task=DLARBittle_PRD_v2 train=DLARBittlePPO_LSTM checkpoint=runs/DLARBittle_GP_0.1-0.8/nn/DLARBittle.pth test=True num_envs=1 torch_deterministic=True

# HB
# python train.py task=DLARBittle_PRD_v2 train=DLARBittlePPO_LSTM checkpoint=runs/DLARBittle_HB_H2_0.3-0.6/nn/DLARBittle.pth test=True num_envs=1 torch_deterministic=True

# PK
# python train.py task=DLARBittle_PRD_v2 train=DLARBittlePPO_LSTM checkpoint=runs/DLARBittle_PK_0.1-0.8/nn/DLARBittle.pth test=True num_envs=1 torch_deterministic=True


python train.py task=DLARBittle_PRD_v2 train=DLARBittlePPO_LSTM checkpoint=runs/DLARBittle/nn/DLARBittle.pth test=True num_envs=1 torch_deterministic=True