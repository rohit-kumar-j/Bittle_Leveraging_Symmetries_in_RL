# Bittle_Leveraging_Symmetries_in_RL


#### About this repository
This repository is the implementation of paper "Leveraging Symmetries in Gaits for Reinforcement Learning: A Case Study on Quadrupedal Gaits", based on [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [Isaac Gym Benchmark Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

Features:
* **Symmetry-based Reward Design for RL**: Incorporate three symmetries (temporal symmetry, time-reversal symmetry and morphological symmetry) into the reward function, and train 4 gaits for a quadrupedal robot [Bittle](https://www.petoi.com/pages/bittle-smart-robot-dog-model-overview).

##### Authors: Jiayu Ding (<jding14@syr.edu>), Xulin Chen (<xchen168@syr.edu>)
##### Affiliation: [DLAR Lab](https://dlarlab.syr.edu/)
##### Instructors: Zhenyu Gan, Garrett E. Katz 

This projected was initially developed at Syracuse University (Dynamic Locomotion and Robotics Lab).

---
#### Publications
This work has been submitted to IROS 2024. If you use this work in an academic context, please cite the following publication: https://arxiv.org/submit/5474477.

---
#### Installation
Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym) and follow the installation instructions. Recommend using an individual conda environment.

Once Isaac Gym is properly installed, download this repository and run the following commands

```
cd Bittle_Leveraging_Symmetries_in_RL/
pip install -e .
pip install -r requirements.txt
```

---
#### Organization of project files
* `cfg/task/DLARBittle_PRD_v2.yaml`: Parameters for creating a Bittle environment.
* `cfg/train/DLARBittlePPO_LSTM.yaml`: The configuration of RL policy training (using PPO algorithms and LSTM network).
* `tasks/dlar_bittle_PRD_v2.py`: The definition of Bittle environment in python.
* `runs/`: Save the trained policies.

---
#### Initialize the environment
Before running any code, change the directory and activate the conda environment
```
cd isaacgymenvs/
conda activate your_conda_env_name
```

---
#### Train a policy
To train policies, run
```
./train.sh
```
The trained policies are saved under `runs/DLARBittle_ww-xx-yy-zz`. In this directory, 
* `nn/DLARBittle.pth` saves the policy parameters achieving the best performance.
* `nn/last_DLARBittle_ep_x_rew_y.pth` files are the policy parameters at training epoch `x` achieving reward `y`.
* `summaries/` includes the log file for training. To visualize the log file, install tensorboard by  run `tensorboard --logdir=/path/to/log/file`.


---
#### Visualize pre-trained policies
To visualize policies, change `checkpoint=/path/to/your/policy` in `visualize.sh`. Then run
```
./visualize.sh
```

We upload 4 pretrained policies for bounding (`runs/DLARBittle_B2_0.1-0.8/`), galloping (`runs/DLARBittle_GP_0.1-0.8/`), half-bounding (`runs/DLARBittle_HB_H2_0.3-0.6/`) and pronking (`runs/DLARBittle_PK_0.1-0.8/`) gait.

---
#### Record video for pre-trained policies
To visualize a policy, change `checkpoint=/path/to/your/policy` in `record_video.sh`. Then run
```
./record_video.sh
```


### Summary of work
https://github.com/DLARlab/Bittle_Leveraging_Symmetries_in_RL/assets/69872954/ab3cce09-828b-4f61-b5ac-0a1872d75d80

>>>>>>> ace04f2ae93e65340a0d10df2a615ed40966767c
