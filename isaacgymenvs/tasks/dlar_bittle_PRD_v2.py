# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import matplotlib.pyplot as plt
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from torch import Tensor
from typing import Tuple, Dict

import math
from collections import defaultdict
import scipy.io


class DLARBittle_PRD_v2(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}

        # 2024.01.24: User improvement
        self.rew_scales['up_scale'] = self.cfg["env"]["learn"]["upScale"]
        self.rew_scales['lin_vel_x_err_coef'] = self.cfg['env']['learn']['lin_vel_x_err_coef']
        self.rew_scales['lin_vel_y_error_coef'] = self.cfg['env']['learn']['lin_vel_y_error_coef']
        self.rew_scales['ang_vel_yaw_error_coef'] = self.cfg['env']['learn']['ang_vel_yaw_error_coef']
        self.rew_scales['E_R_cmd_coef'] = self.cfg['env']['learn']['E_R_cmd_coef']
        self.rew_scales['torque_diff_coef'] = self.cfg['env']['learn']['torque_diff_coef']
        self.rew_scales['E_R_smooth_coef'] = self.cfg['env']['learn']['E_R_smooth_coef']

        self.rew_scales['prd_contact_force_coef'] = self.cfg['env']['learn']['prd_contact_force_coef']
        self.rew_scales['E_C_frc_val_offset'] = self.cfg['env']['learn']['E_C_frc_val_offset']
        self.rew_scales['prd_velocity_coef'] = self.cfg['env']['learn']['prd_velocity_coef']
        self.rew_scales['E_C_spd_val_offset'] = self.cfg['env']['learn']['E_C_spd_val_offset']
        self.rew_scales['E_R_prd_coef'] = self.cfg['env']['learn']['E_R_prd_coef']

        self.rew_scales['morphological_kinematic_diff_coef'] = self.cfg['env']['learn']['morphological_kinematic_diff_coef']
        self.rew_scales['morphological_sigma'] = self.cfg['env']['learn']['morphological_sigma']
        self.rew_scales['E_R_morphological_kinematic_coef'] = self.cfg['env']['learn']['E_R_morphological_kinematic_coef']

        # self.rew_scales['bounding_kinematic_diff_coef'] = self.cfg['env']['learn']['bounding_kinematic_diff_coef']
        # self.rew_scales['bounding_dynamic_diff_coef'] = self.cfg['env']['learn']['bounding_dynamic_diff_coef']
        # self.rew_scales['E_R_bounding_kinematic_coef'] = self.cfg['env']['learn']['E_R_bounding_kinematic_coef']
        # self.rew_scales['E_R_bounding_dynamic_coef'] = self.cfg['env']['learn']['E_R_bounding_dynamic_coef']

        self.rew_scales['pitching_rew_coef'] = self.cfg['env']['learn']['pitching_rew_coef']
        self.rew_scales['E_R_pitching_coef'] = self.cfg['env']['learn']['E_R_pitching_coef']

        # Velocity command
        self.commands_dict = {}
        self.commands_dict['target_lin_vel_x'] = self.cfg['env']['periodicalRewardComposition']['target_lin_vel_x']
        self.commands_dict['target_lin_vel_x_scale_range'] = self.cfg['env']['periodicalRewardComposition']['target_lin_vel_x_scale_range']
        self.commands_dict['target_lin_vel_y'] = self.cfg['env']['periodicalRewardComposition']['target_lin_vel_y']

        self.commands_dict['target_lin_vel_y_scale_range_sign'] = self.cfg['env']['periodicalRewardComposition']['target_lin_vel_y_scale_range_sign']
        self.commands_dict['target_lin_vel_y_scale_range_scale'] = self.cfg['env']['periodicalRewardComposition']['target_lin_vel_y_scale_range_scale']

        self.commands_dict['target_ang_vel_yaw'] = self.cfg['env']['periodicalRewardComposition']['target_ang_vel_yaw']
        self.commands_dict['target_ang_vel_yaw_scale_range'] = self.cfg['env']['periodicalRewardComposition']['target_ang_vel_yaw_scale_range']

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 32
        self.cfg["env"]["numActions"] = 8

        print(self.cfg)
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # # 2024.01.24: fuck
        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        '''
            2023.12.03: Get the tensor of rigid body states
        '''
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        # print('Base height: ', torch.mean(self.root_states[:, 2]))
        # print('Foot height: ', torch.mean(self.rigid_body_state[:, 8, 2]))
        # print('Foot height: ', torch.mean(self.rigid_body_state[:, 5, 2]))
        # print('Foot height: ', torch.mean(self.rigid_body_state[:, 16, 2]))
        # print('Foot height: ', torch.mean(self.rigid_body_state[:, 13, 2]))

        # Velocity command buffer
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_linvel_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_linvel_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_angvel_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        # period(stride time)
        self.period = self.cfg['env']['periodicalRewardComposition']['period']
        self.period_scale_range = self.cfg['env']['periodicalRewardComposition']['period_scale_range']
        self.nop = self.cfg["env"]["periodicalRewardComposition"]["episodeLength_nop"]
        self.period_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.num_steps_one_period_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # duty factor(stance time/stride time)
        self.duty_factor = self.cfg['env']['periodicalRewardComposition']['duty_factor']
        self.duty_factor_scale_range = self.cfg['env']['periodicalRewardComposition']['duty_factor_scale_range']
        self.duty_factor_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Initial position
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # 2024.02.12: theta for the reversed velocity
        # self.theta_left_front_reverse = 1 - self.theta_left_front - self.duty_factor
        # self.theta_left_rear_reverse = 1 - self.theta_left_rear - self.duty_factor
        # self.theta_right_front_reverse = 1 - self.theta_right_front - self.duty_factor
        # self.theta_right_rear_reverse = 1 - self.theta_right_rear - self.duty_factor
        theta_left_front = self.cfg["env"]["periodicalRewardComposition"]["theta_left_front"]
        theta_left_rear = self.cfg["env"]["periodicalRewardComposition"]["theta_left_rear"]
        theta_right_front = self.cfg["env"]["periodicalRewardComposition"]["theta_right_front"]
        theta_right_rear = self.cfg["env"]["periodicalRewardComposition"]["theta_right_rear"]

        self.theta_front_random_range = self.cfg["env"]["periodicalRewardComposition"]["theta_front_random_range"]
        self.theta_rear_random_range = self.cfg["env"]["periodicalRewardComposition"]["theta_rear_random_range"]

        # 2024.02.21: theta should change with respect to duty factor
        positive_vel_thetas = torch.tensor([theta_left_front, theta_left_rear, theta_right_front, theta_right_rear], dtype = torch.float, device = self.device, requires_grad = False)
        # negative_vel_thetas = 1 - self.duty_factor - positive_vel_thetas
        # self.init_thetas = torch.stack([positive_vel_thetas, negative_vel_thetas], dim = 0) # [2, 4]
        self.init_thetas = positive_vel_thetas.unsqueeze(0)
        self.theta_buffer = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # LF, LB, RF, RB

        self.c_swing_frc = self.cfg["env"]["periodicalRewardComposition"]["c_swing_frc"]
        self.c_swing_spd = self.cfg["env"]["periodicalRewardComposition"]["c_swing_spd"]
        self.c_stance_frc = self.cfg["env"]["periodicalRewardComposition"]["c_stance_frc"]
        self.c_stance_spd = self.cfg["env"]["periodicalRewardComposition"]["c_stance_spd"]
        # self.swing_start = self.cfg["env"]["periodicalRewardComposition"]["swing_start"]
        # self.swing_end = self.cfg["env"]["periodicalRewardComposition"]["swing_end"]
        # self.stance_start = self.cfg["env"]["periodicalRewardComposition"]["stance_start"]
        # self.stance_end = self.cfg["env"]["periodicalRewardComposition"]["stance_end"]
        self.kappa = self.cfg["env"]["periodicalRewardComposition"]["kappa"]

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 2024.02.24: Save the state at last second
        self.last_root_states = torch.zeros_like(self.root_states)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()        
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))

        # Print the information of DoFs
        # {
        #   'left-back-shoulder-joint': 0, 
        #   'left-back-knee-joint': 1, 
        #   'left-front-shoulder-joint': 2, 
        #   'left-front-knee-joint': 3, 
        #   'right-back-shoulder-joint': 4, 
        #   'right-back-knee-joint': 5, 
        #   'right-front-shoulder-joint': 6
        #   'right-front-knee-joint': 7, 
        # }
        # {'observation_space': Box(-inf, inf, (23,), float32), 'action_space': Box(-1.0, 1.0, (8,), float32), 'agents': 1, 'value_size': 1}
        print(self.gym.get_actor_dof_dict(self.envs[0], 0))
        print(self.contact_forces.size())

        # Print the information of rigid bodies
        # 2023.12.06: add 4 foot sole links
        # {
        #   'base-frame-link': 0, 
        #   'battery-link': 1, 
        #   'cover-link': 2, 
        #   'left-back-shoulder-link': 3, 
        #   'left-back-knee-link': 4, 
        #   'left-back-foot-sole-link': 5, 
        #   'left-front-shoulder-link': 6, 
        #   'left-front-knee-link': 7, 
        #   'left-front-foot-sole-link': 8, 
        #   'mainboard_link': 9, 
        #   'imu_link': 10, 
        #   'right-back-shoulder-link': 11, 
        #   'right-back-knee-link': 12, 
        #   'right-back-foot-sole-link': 13, 
        #   'right-front-shoulder-link': 14,
        #   'right-front-knee-link': 15, 
        #   'right-front-foot-sole-link': 16, 
        # }
        print(self.gym.get_actor_rigid_body_dict(self.envs[0], 0))

        # 2023.11.15
        # self.num_steps_one_period = int(self.cfg["env"]["periodicalRewardComposition"]["period"] / self.sim_params.dt)
        # Set visualize to False if training
        # self.create_buffer_for_joint_visualization(visualize = False)


        # Periodical reward composition
        # self.period = self.cfg["env"]["periodicalRewardComposition"]["period"]
        self.dt = self.sim_params.dt
        # self.nop = self.cfg["env"]["periodicalRewardComposition"]["episodeLength_nop"]
        # self.max_episode_length = int(self.nop * self.period / self.dt)
        # self.num_steps_one_period = int(self.period / self.dt)
        # self.theta_left_front = self.cfg["env"]["periodicalRewardComposition"]["theta_left_front"]
        # self.theta_left_rear = self.cfg["env"]["periodicalRewardComposition"]["theta_left_rear"]
        # self.theta_right_front = self.cfg["env"]["periodicalRewardComposition"]["theta_right_front"]
        # self.theta_right_rear = self.cfg["env"]["periodicalRewardComposition"]["theta_right_rear"]


        # 2023.12.08
        self.is_plot_foot_periodicity = False
        # Visualization function. Uncomment the following line before evaluation.
        # self.create_foot_periodicity_visualization(env_idx = 0)

    # 2023.12.08
    def create_foot_periodicity_visualization(self, env_idx = 0):
        self.is_plot_foot_periodicity = True
        self.plot_env_idx = env_idx
        self.plot_data = defaultdict(lambda: [])
        self.num_collected_plot_data = 0


    def get_foot_periodicity(self):
        if self.is_plot_foot_periodicity == True:
            if self.num_collected_plot_data < self.max_episode_length[self.plot_env_idx]:
                E_C_frc_lf, E_C_frc_lb, E_C_frc_rf, E_C_frc_rb, E_C_spd_lf, E_C_spd_lb, E_C_spd_rf, E_C_spd_rb = self._compute_E_C()

                self.plot_data['foot_grf'].append(torch.stack([
                    torch.norm(self.contact_forces[self.plot_env_idx, 7, :], dim = -1), # lf_frc
                    torch.norm(self.contact_forces[self.plot_env_idx, 4, :], dim = -1), # lb_frc
                    torch.norm(self.contact_forces[self.plot_env_idx, 15, :], dim = -1), # rf_frc
                    torch.norm(self.contact_forces[self.plot_env_idx, 12, :], dim = -1), # rb_frc
                ]).detach().cpu().numpy())

                self.plot_data['E_C_frc'].append(torch.stack([
                    E_C_frc_lf[self.plot_env_idx], 
                    E_C_frc_lb[self.plot_env_idx], 
                    E_C_frc_rf[self.plot_env_idx], 
                    E_C_frc_rb[self.plot_env_idx],
                ]).detach().cpu().numpy())
            
                self.plot_data['foot_spd'].append(torch.stack([
                    torch.norm(self.rigid_body_state[self.plot_env_idx, 8, 7:10], dim = -1), # lf_linvel
                    torch.norm(self.rigid_body_state[self.plot_env_idx, 5, 7:10], dim = -1), # lb_linvel
                    torch.norm(self.rigid_body_state[self.plot_env_idx, 16, 7:10], dim = -1), # rf_linvel
                    torch.norm(self.rigid_body_state[self.plot_env_idx, 13, 7:10], dim = -1), # rb_linvel
                ]).detach().cpu().numpy())

                self.plot_data['E_C_spd'].append(torch.stack([
                    E_C_spd_lf[self.plot_env_idx], 
                    E_C_spd_lb[self.plot_env_idx], 
                    E_C_spd_rf[self.plot_env_idx], 
                    E_C_spd_rb[self.plot_env_idx],
                ]).detach().cpu().numpy())

                # 2024.02.21: Add command / true linear velocity on y-axis
                base_quat = self.root_states[[self.plot_env_idx], 3:7]
                base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[[self.plot_env_idx], 7:10]).squeeze(0)

                self.plot_data['command_y_linvel'].append(self.commands_linvel_y[self.plot_env_idx].detach().cpu().numpy())
                self.plot_data['true_y_linvel'].append(base_lin_vel[1].detach().cpu().numpy())

                # 2024.02.21: Add command / true torso pitch angle
                pitching_rew_coef = (E_C_frc_lf + E_C_frc_rf - E_C_frc_lb - E_C_frc_rb)/2
                self.plot_data['pitching_rew_coef'].append(pitching_rew_coef[self.plot_env_idx].detach().cpu().numpy())
                curr_pitch_ang_vel = quat_rotate_inverse(base_quat, self.root_states[[self.plot_env_idx], 10:13])
                last_base_quat = self.last_root_states[[self.plot_env_idx], 3:7]
                last_pitch_ang_vel = quat_rotate_inverse(last_base_quat, self.last_root_states[[self.plot_env_idx], 10:13])
                pitch_acc = (curr_pitch_ang_vel - last_pitch_ang_vel)[0, 0] / self.dt
                self.plot_data['true_base_pitch_acc'].append(pitch_acc.detach().cpu().numpy())
                # print(base_ang_vel[[self.plot_env_idx], 1].detach().cpu().numpy())

                self.plot_data['duty_factor'].append(self.duty_factor_buffer[self.plot_env_idx].detach().cpu().numpy())
                self.plot_data['theta'].append(self.theta_buffer[self.plot_env_idx].detach().cpu().numpy())
                self.plot_data['commands_linvel_y'].append(self.commands_linvel_y[self.plot_env_idx].detach().cpu().numpy())
                self.plot_data['period'].append(self.period_buffer[self.plot_env_idx].detach().cpu().numpy())

                self.num_collected_plot_data += 1




    def plot_foot_periodicity(self):
        if self.is_plot_foot_periodicity == True:
            # 2024.02.27: Only keep the data in certain periods
            num_steps_per_period = self.num_steps_one_period_buffer[self.plot_env_idx].int().detach().cpu().numpy()
            start_idx = 4 * num_steps_per_period
            end_idx = 8 * num_steps_per_period
            print('num_steps_per_period:', num_steps_per_period)
            print(f'time step range: [{start_idx}, {end_idx})')

            # 2024.01.25: Save the data instead, then run plot_periodic_data.py to plot figures. 
            for key, data in self.plot_data.items():
                print(key)
                self.plot_data[key] = np.array(data)[start_idx:end_idx]
                # print(key, self.plot_data[key].shape)
            self.plot_data = dict(self.plot_data)
            self.plot_data['plot_idx_range'] = np.array([start_idx, end_idx])
            self.plot_data['num_steps_per_period'] = num_steps_per_period
            self.plot_data['dt'] = self.dt
            # np.save('periodic_data.npy', self.plot_data, allow_pickle = True)
            scipy.io.savemat('periodic_data.mat', self.plot_data)


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        # 2024.1.16
        # bittle_PRD_v1 modified from the bittle.urdf online is in correct in the scale of dimensions and inertia
        # bittle_PRD_v2: scale the dimensions by 1/10 and inertia by 1/100
        asset_file = "urdf/bittle/urdf/bittle_PRD_v2.urdf"
        #asset_path = os.path.join(asset_root, asset_file)
        #asset_root = os.path.dirname(asset_path)
        #asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        #asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        #asset_options.collapse_fixed_joints = True
        #asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        #asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.4
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        start_pose = gymapi.Transform()
        start_pose.r = gymapi.Quat.from_euler_zyx(*self.base_init_state[3:6])
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "knee"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "shoulder" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] #gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "anymal", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base-frame-link")

    def pre_physics_step(self, actions):
        # 2024.02.24: Save the state at last second
        self._save_last_root_states()

        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.prev_torques = self.torques.clone()
        self.compute_observations()
        self.compute_reward(self.actions)

        # 2023.11.15: Fix the position of camera
        self._update_camera()

        self.get_foot_periodicity()


    # Focus on one bittle, making it more convenient to record videos.
    def _update_camera(self):
        env0_root_pos = self.root_states[0, :3]
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], gymapi.Vec3(env0_root_pos[0] + 0.4, env0_root_pos[1], env0_root_pos[2]), gymapi.Vec3(*env0_root_pos))


    def _compute_E_C(self):
        phi = (self.progress_buf / self.num_steps_one_period_buffer).cpu().numpy()
        duty_factor_buffer_np = self.duty_factor_buffer.cpu().numpy()
        theta_buffer_np = self.theta_buffer.cpu().numpy()
        E_C_frc_left_front_val = torch.from_numpy(E_C_frc(phi, duty_factor_buffer_np, self.kappa, self.c_swing_frc, self.c_stance_frc, theta_buffer_np[:, 0])).to(self.device)
        E_C_frc_left_back_val = torch.from_numpy(E_C_frc(phi, duty_factor_buffer_np, self.kappa, self.c_swing_frc, self.c_stance_frc, theta_buffer_np[:, 1])).to(self.device)
        E_C_frc_right_front_val = torch.from_numpy(E_C_frc(phi, duty_factor_buffer_np, self.kappa, self.c_swing_frc, self.c_stance_frc, theta_buffer_np[:, 2])).to(self.device)
        E_C_frc_right_back_val = torch.from_numpy(E_C_frc(phi, duty_factor_buffer_np, self.kappa, self.c_swing_frc, self.c_stance_frc, theta_buffer_np[:, 3])).to(self.device)
        E_C_spd_left_front_val = torch.from_numpy(E_C_spd(phi, duty_factor_buffer_np, self.kappa, self.c_swing_spd, self.c_stance_spd, theta_buffer_np[:, 0])).to(self.device)
        E_C_spd_left_back_val = torch.from_numpy(E_C_spd(phi, duty_factor_buffer_np, self.kappa, self.c_swing_spd, self.c_stance_spd, theta_buffer_np[:, 1])).to(self.device)
        E_C_spd_right_front_val = torch.from_numpy(E_C_spd(phi, duty_factor_buffer_np, self.kappa, self.c_swing_spd, self.c_stance_spd, theta_buffer_np[:, 2])).to(self.device)
        E_C_spd_right_back_val = torch.from_numpy(E_C_spd(phi, duty_factor_buffer_np, self.kappa, self.c_swing_spd, self.c_stance_spd, theta_buffer_np[:, 3])).to(self.device)
        return E_C_frc_left_front_val, E_C_frc_left_back_val, E_C_frc_right_front_val, E_C_frc_right_back_val, E_C_spd_left_front_val, E_C_spd_left_back_val, E_C_spd_right_front_val, E_C_spd_right_back_val


    def compute_reward(self, actions):
        

        self.rew_buf[:], self.reset_buf[:] = compute_anymal_reward(
            # tensors
            self.root_states,
            self.last_root_states,
            # self.commands,
            self.prev_torques,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            self.dof_pos,
            self.rigid_body_state,
            # 2024.01.29
            self.commands_linvel_x,
            self.commands_linvel_y,
            self.commands_angvel_yaw,
            # 2024.01.09
            self.num_steps_one_period_buffer,
            self.max_episode_length,
            self.theta_buffer,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.targets,
            self.heading_vec,
            self.up_vec,
            self.inv_start_rot,
            # 2023.12.03
            *self._compute_E_C(),
            self.dt
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf[:] = compute_anymal_observations(  # tensors
                                                        self.root_states,
                                                        # self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        self.progress_buf,
                                                        self.rigid_body_state,
                                                        # 2024.01.29: Add commands
                                                        self.commands_linvel_x,
                                                        self.commands_linvel_y,
                                                        self.commands_angvel_yaw,
                                                        # 2024.01.29: Add period and duty factors
                                                        self.period_buffer,
                                                        self.max_episode_length,
                                                        self.num_steps_one_period_buffer,
                                                        self.duty_factor_buffer,
                                                        # 2024.02.12: Theta buffer
                                                        self.theta_buffer,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        # self.theta_left_front,
                                                        # self.theta_left_rear,
                                                        # self.theta_right_front,
                                                        # self.theta_right_rear,
        )

        # print(self.root_states.size(), self.rigid_body_state.size())
        # print(self.num_bodies)

        # # Print the information
        # lf_knee_cf = self.contact_forces[0, 7, :]
        # lb_knee_cf = self.contact_forces[0, 4, :]
        # rf_knee_cf = self.contact_forces[0, 15, :]
        # rb_knee_cf = self.contact_forces[0, 12, :]
        # print(f'CF: LF_knee: {lf_knee_cf} | LB_knee: {lb_knee_cf} | RF_knee: {rf_knee_cf} | RB_knee: {rb_knee_cf}')
        # # print(self.root_states[0, 2])  

        # base_quat = self.root_states[:, 3:7]
        # base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10]) # Base linear velocity in world coordinate
        # print(base_lin_vel[0, 1])

        # Print the base height
        # print(self.root_states[0, 2])


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # Reset velocity command
        # liner_x_velocity
        commands_linvel_x_scale = torch_rand_float(self.commands_dict['target_lin_vel_x_scale_range'][0], self.commands_dict['target_lin_vel_x_scale_range'][1], (len(env_ids), 1), device = self.device).squeeze()
        self.commands_linvel_x[env_ids] = self.commands_dict['target_lin_vel_x'] * commands_linvel_x_scale
        # commands_linvel_y_scale = torch_rand_float(self.commands_dict['target_lin_vel_y_scale_range'][0], self.commands_dict['target_lin_vel_y_scale_range'][1], (len(env_ids), 1), device = self.device).squeeze()
        
        # liner_y_velocity
        # 2024.02.12: Changet the range to a set of specific values
        commands_linvel_y_range_sign = torch.tensor(self.commands_dict['target_lin_vel_y_scale_range_sign'], device = self.progress_buf.device)[torch.randint(high = len(self.commands_dict['target_lin_vel_y_scale_range_sign']), size = (len(env_ids),))]
        commands_linvel_y_scale = torch_rand_float(self.commands_dict['target_lin_vel_y_scale_range_scale'][0], self.commands_dict['target_lin_vel_y_scale_range_scale'][1], (len(env_ids), 1), device = self.device).squeeze()
        self.commands_linvel_y[env_ids] = self.commands_dict['target_lin_vel_y'] * commands_linvel_y_range_sign * commands_linvel_y_scale

        # angular_yaw_velocity
        commands_angvel_yaw_scale = torch_rand_float(self.commands_dict['target_ang_vel_yaw_scale_range'][0], self.commands_dict['target_ang_vel_yaw_scale_range'][1], (len(env_ids), 1), device = self.device).squeeze()
        self.commands_angvel_yaw[env_ids] = self.commands_dict['target_ang_vel_yaw'] * commands_angvel_yaw_scale

        
        # 2024.02.12: Reset theta buffer
        # positive_commands_linvel_y_env_inds = (self.commands_linvel_y[env_ids] < 0).long()
        # self.theta_buffer[env_ids] = self.init_thetas[positive_commands_linvel_y_env_inds]
        # self.theta_buffer[env_ids, :] = self.init_thetas[0].unsqueeze(0)


        self.theta_buffer[env_ids, :] = self.init_thetas
        # # # 2024.3.14: Reset CMD velocity within range based on theta_F-theta_B
        # theta_F = 0.5 * (self.theta_buffer[env_ids, 0] + self.theta_buffer[env_ids, 2])
        # theta_B = 0.5 * (self.theta_buffer[env_ids, 1] + self.theta_buffer[env_ids, 3])
        # theta_diff = theta_F - theta_B
        # lower_limit = -0.00016 * torch.exp(13.01 * theta_diff) + 0.2956 * torch.exp(-0.9554 * theta_diff)
        # upper_limit = 0.3054 * torch.exp(-0.742 * theta_diff) + 0.02192 * torch.exp(6.584 * theta_diff)
        # self.commands_linvel_y[env_ids] = torch.rand_like(lower_limit) * (upper_limit - lower_limit) + lower_limit
        # self.commands_linvel_y[env_ids] *= commands_linvel_y_range_sign 

        
        # # Reset theta values if the linear velocity is negative
        negative_command_linvel_y_env_inds = env_ids[torch.where(self.commands_linvel_y[env_ids] < 0)[0]]
        self.theta_buffer[negative_command_linvel_y_env_inds, :] *= -1
        self.theta_buffer[negative_command_linvel_y_env_inds, :] += 1 - self.duty_factor_buffer[negative_command_linvel_y_env_inds].unsqueeze(-1)

        front_random = torch_rand_float(*self.theta_front_random_range, shape = (len(env_ids), 1), device = self.device).squeeze()
        self.theta_buffer[env_ids, 0] -= front_random
        self.theta_buffer[env_ids, 2] += front_random

        rear_random = torch_rand_float(*self.theta_rear_random_range, shape = (len(env_ids), 1), device = self.device).squeeze()
        self.theta_buffer[env_ids, 1] += rear_random
        self.theta_buffer[env_ids, 3] -= rear_random

        # # 2024.3.4: added opposite phase shift to left/right
        # rear_random = torch_rand_float(*self.theta_rear_random_range, shape = (len(env_ids), 1), device = self.device).squeeze()
        # self.theta_buffer[env_ids, 1] += rear_random
        # self.theta_buffer[env_ids, 3] -= rear_random


            


        
        
        # 2024.02.21: Reset the period and duty factor given the command linear y velocity
        env_command_linvel_absys = torch.abs(self.commands_linvel_y[env_ids])
        scale_coef1 = torch.rand_like(env_ids.float()) * 2 - 1.0
        self.period_buffer[env_ids] = 0.2576 * torch.exp(-0.9829 * env_command_linvel_absys) * (1 + scale_coef1 * env_command_linvel_absys / 4)
        self.max_episode_length[env_ids] = self.period_buffer[env_ids] * self.nop / self.dt
        self.num_steps_one_period_buffer[env_ids] = self.period_buffer[env_ids] / self.dt
        scale_coef2 = torch.rand_like(env_ids.float()) * 2 - 1.0
        # self.duty_factor_buffer[env_ids] = 0.5388 * torch.exp(-0.7075 * torch.abs(self.commands_linvel_y[env_ids])) * (1 + scale_coef2 * env_command_linvel_absys / 2.5)
        self.duty_factor_buffer[env_ids] = 0.5588 * torch.exp(-0.6875 * torch.abs(self.commands_linvel_y[env_ids])) * (1 + scale_coef2 * env_command_linvel_absys / 4)




        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # # # print out random values
        # print('velocity_y', self.commands_linvel_y)
        # print('period', self.period_buffer)
        # print('duty factor', self.duty_factor_buffer)
        # print('theta', self.theta_buffer)

        # 2024.02.24
        self.last_root_states[env_ids] = self.root_states[env_ids].clone()


    def __del__(self):
        self.plot_foot_periodicity()


    # 2024.02.24: Save the state at last second
    def _save_last_root_states(self):
        self.last_root_states[:] = self.root_states.clone()



#####################################################################
###=========================jit functions=========================###
#####################################################################


# 2023.11.07: Customize the reward function
@torch.jit.script
def compute_anymal_reward(
    # tensors
    root_states: Tensor,
    last_root_states : Tensor,
    # commands: Tensor,
    prev_torques: Tensor,
    torques: Tensor,
    contact_forces: Tensor,
    knee_indices: Tensor,
    episode_lengths: Tensor,
    dof_pos : Tensor,
    rigid_body_state : Tensor,
    # 2024.01.29
    commands_linvel_x : Tensor,
    commands_linvel_y : Tensor,
    commands_angvel_yaw : Tensor,
    # 2024.01.29
    num_steps_one_period : Tensor,
    max_episode_length: Tensor,
    theta_buffer : Tensor,
    # Dict
    rew_scales: Dict[str, float],
    # other
    base_index: int,
    targets,
    vec0,
    vec1,
    inv_start_rot,
    # # 2023.12.03
    E_C_frc_lf_val,
    E_C_frc_lb_val,
    E_C_frc_rf_val,
    E_C_frc_rb_val,
    E_C_spd_lf_val,
    E_C_spd_lb_val,
    E_C_spd_rf_val,
    E_C_spd_rb_val,
    dt : float,
) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) # Base linear velocity in world coordinate
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]

    to_target = targets - torso_position
    to_target[:, 2] = 0
    # target_dirs = normalize(to_target)

    num_envs = torso_position.shape[0]
    torso_quat = quat_mul(torso_rotation, inv_start_rot)

    # heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    # heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)
    # # reward from direction headed
    # heading_weight_tensor = torch.ones_like(heading_proj) * rew_scales["heading_scale"]
    # heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, rew_scales["heading_scale"] * heading_proj / 0.8)

    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    up_proj = up_vec[:, 2]

    # aligning up axis of ant and environment (alive reward)
    up_reward = torch.zeros_like(up_proj)
    # up_reward = torch.where(up_proj > 0.9, up_reward + rew_scales['up_scale'], up_reward)
    up_reward = torch.where(up_proj > 0.85, up_reward + rew_scales['up_scale'], up_reward)
    
    # # 2024.1.11: Sigmoid function to smoothly add soft constraints
    # # Spend few strides to move from the initial state to the periodical solution set
    # warmup_coef1 = torch.sigmoid(3 * (episode_lengths / num_steps_one_period - 20.0))

    # # 2024.1.11: Sigmoid function to smoothly add soft constraints
    # # Spend few strides to move from the initial state to the periodical solution set
    # warmup_coef2 = torch.sigmoid(3 * (episode_lengths / num_steps_one_period - 30.0))

    
    '''
        Command reward
    '''
    # Penalize drifting
    lin_vel_x_error = torch.abs(base_lin_vel[:, 0] - commands_linvel_x)
    rew_lin_vel_x = -1 * (1 - torch.exp(-rew_scales['lin_vel_x_err_coef'] * lin_vel_x_error))
    
    # Encourage moving forward
    # Based on the prediction of SLIP model, the period should be 0.4s for the velocity of 2.5 m/s
    lin_vel_y_error = torch.abs(base_lin_vel[:, 1] - commands_linvel_y)
    rew_lin_vel_y = -1 * (1 - torch.exp(-rew_scales['lin_vel_y_error_coef'] * lin_vel_y_error))

    # Penalize yaw velocity
    ang_vel_yaw_error = torch.abs(base_ang_vel[:, 2] - commands_angvel_yaw)
    rew_ang_vel_yaw = -1 * (1 - torch.exp(-rew_scales['ang_vel_yaw_error_coef'] * ang_vel_yaw_error))

    # Command reward
    E_R_cmd = rew_scales['E_R_cmd_coef'] * (rew_lin_vel_x + rew_lin_vel_y + rew_ang_vel_yaw)
    # print(E_R_cmd)

    '''
        Smoothness reward
    '''
    # Torque difference between two successive time steps
    torque_diff = torch.sum(torch.abs(prev_torques - torques), dim = -1)
    rew_torque_diff = -1 * (1 - torch.exp(-rew_scales['torque_diff_coef'] * torque_diff))

    # Smoothness reward
    E_R_smooth = rew_scales['E_R_smooth_coef'] * rew_torque_diff

    # print('torques', torques)


    '''
        2023.12.03: Periodical reward composition
    '''
    # 2023.12.20: Acquire the contact force of the knee links
    lf_cf = contact_forces[:, 7, :] 
    lf_cf_norm = torch.norm(lf_cf, dim = -1)
    E_R_frc_lf = (E_C_frc_lf_val + rew_scales['E_C_frc_val_offset']) * (1 - torch.exp(-rew_scales['prd_contact_force_coef'] * lf_cf_norm**2))

    lb_cf = contact_forces[:, 4, :]
    lb_cf_norm = torch.norm(lb_cf, dim = -1)
    E_R_frc_lb = (E_C_frc_lb_val + rew_scales['E_C_frc_val_offset']) * (1 - torch.exp(-rew_scales['prd_contact_force_coef'] * lb_cf_norm**2))

    rf_cf = contact_forces[:, 15, :] 
    rf_cf_norm = torch.norm(rf_cf, dim = -1)
    E_R_frc_rf = (E_C_frc_rf_val + rew_scales['E_C_frc_val_offset']) * (1 - torch.exp(-rew_scales['prd_contact_force_coef'] * rf_cf_norm**2))

    rb_cf = contact_forces[:, 12, :]
    rb_cf_norm = torch.norm(rb_cf, dim = -1)
    E_R_frc_rb = (E_C_frc_rb_val + rew_scales['E_C_frc_val_offset']) * (1 - torch.exp(-rew_scales['prd_contact_force_coef'] * rb_cf_norm**2))

    E_R_frc = E_R_frc_lf + E_R_frc_lb + E_R_frc_rf + E_R_frc_rb


    # Foot velocities: Acquire the velocity of sole links
    lf_linvel = rigid_body_state[:, 8, 7:10]
    lf_linvel_norm = torch.norm(lf_linvel, dim = -1)
    E_R_spd_lf = (E_C_spd_lf_val + rew_scales['E_C_spd_val_offset']) * (1 - torch.exp(-rew_scales['prd_velocity_coef'] * lf_linvel_norm**2))

    lb_linvel = rigid_body_state[:, 5, 7:10]
    lb_linvel_norm = torch.norm(lb_linvel, dim = -1)
    E_R_spd_lb = (E_C_spd_lb_val + rew_scales['E_C_spd_val_offset']) * (1 - torch.exp(-rew_scales['prd_velocity_coef'] * lb_linvel_norm**2))

    rf_linvel = rigid_body_state[:, 16, 7:10]
    rf_linvel_norm = torch.norm(rf_linvel, dim = -1)
    E_R_spd_rf = (E_C_spd_rf_val + rew_scales['E_C_spd_val_offset']) * (1 - torch.exp(-rew_scales['prd_velocity_coef'] * rf_linvel_norm**2))

    rb_linvel = rigid_body_state[:, 13, 7:10]
    rb_linvel_norm = torch.norm(rb_linvel, dim = -1)
    E_R_spd_rb = (E_C_spd_rb_val + rew_scales['E_C_spd_val_offset']) * (1 - torch.exp(-rew_scales['prd_velocity_coef'] * rb_linvel_norm**2))

    E_R_spd = E_R_spd_lf + E_R_spd_lb + E_R_spd_rf + E_R_spd_rb

    # Periodical reward composition (negative, since E_C variables are not greater than 0)
    # E_R_quad_periodical =  0.5 * warmup_coef1 *(E_R_frc + E_R_spd)
    E_R_Prd =  rew_scales['E_R_prd_coef'] * (E_R_frc + E_R_spd)
    


    # # User Improvement
    # #    2024.1.8: Symmetry constraints

    # #    Joint indices: 
    # #        from get_actor_dof_dict()
    # # {
    # #  'left-back-shoulder-joint': 0, 
    # #  'left-back-knee-joint': 1, 
    # #  'left-front-shoulder-joint': 2, 
    # #  'left-front-knee-joint': 3, 
    # #  'right-back-shoulder-joint': 4, 
    # #  'right-back-knee-joint': 5, 
    # #  'right-front-shoulder-joint': 6
    # #  'right-front-knee-joint': 7, 
    # # }

    #     dof_pos shape: [num_envs, num_dofs]
    # '''

    # 2024.2.23: general morphological symmetry decomposition
    # theta_buffer: LF, LB, RF, RB
    morpho_sigma = rew_scales['morphological_sigma']
    morpho_kine_coef = rew_scales['morphological_kinematic_diff_coef']
    # front legs left/right symmetry
    kine_diff_lf_rf_shoulder = torch.exp(-0.5 * ((dof_pos[:, 2] + dof_pos[:, 6]) / morpho_sigma)**2 )
    r_lf_rf_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_rf_shoulder))
    kine_diff_lf_rf_knee = torch.exp(-0.5 * ((dof_pos[:, 3] + dof_pos[:, 7]) / morpho_sigma)**2 )
    r_lf_rf_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_rf_knee))
    E_R_lf_rf = (theta_buffer[:, 0] == theta_buffer[:, 2]) * (r_lf_rf_s + r_lf_rf_k)
    # back legs left/right symmetry
    kine_diff_lb_rb_shoulder = torch.exp(-0.5 * ((dof_pos[:, 0] + dof_pos[:, 4]) / morpho_sigma)**2 )
    r_lb_rb_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lb_rb_shoulder))
    kine_diff_lb_rb_knee = torch.exp(-0.5 * ((dof_pos[:, 1] + dof_pos[:, 5]) / morpho_sigma)**2 )
    r_lb_rb_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lb_rb_knee))
    E_R_lb_rb = (theta_buffer[:, 1] == theta_buffer[:, 3]) * (r_lb_rb_s + r_lb_rb_k)


    # left legs front/back symmetry
    kine_diff_lf_lb_shoulder = torch.exp(-0.5 * ((dof_pos[:, 0] - dof_pos[:, 2]) / morpho_sigma)**2 )
    r_lf_lb_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_lb_shoulder))
    kine_diff_lf_lb_knee = torch.exp(-0.5 * ((dof_pos[:, 1] - dof_pos[:, 3]) / morpho_sigma)**2 )
    r_lf_lb_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_lb_knee))
    E_R_lf_lb = (theta_buffer[:, 0] == theta_buffer[:, 1]) * (r_lf_lb_s + r_lf_lb_k)
    # right legs front/back symmetry
    kine_diff_rf_rb_shoulder = torch.exp(-0.5 * ((dof_pos[:, 4] - dof_pos[:, 6]) / morpho_sigma)**2 )
    r_rf_rb_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_rf_rb_shoulder))
    kine_diff_rf_rb_knee = torch.exp(-0.5 * ((dof_pos[:, 5] - dof_pos[:, 7]) / morpho_sigma)**2 )
    r_rf_rb_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_rf_rb_knee))
    E_R_rf_rb = (theta_buffer[:, 2] == theta_buffer[:, 3]) * (r_rf_rb_s + r_rf_rb_k)

    # diagonal leg symmetry
    # front left / back right symmetry
    kine_diff_lf_rb_shoulder = torch.exp(-0.5 * ((dof_pos[:, 2] + dof_pos[:, 4]) / morpho_sigma)**2 )
    r_lf_rb_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_rb_shoulder))
    kine_diff_lf_rb_knee = torch.exp(-0.5 * ((dof_pos[:, 3] + dof_pos[:, 5]) / morpho_sigma)**2 )
    r_lf_rb_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_lf_rb_knee))
    E_R_lf_rb = (theta_buffer[:, 0] == theta_buffer[:, 3]) * (r_lf_rb_s + r_lf_rb_k)
    # front right / left back symmetry
    kine_diff_rf_lb_shoulder = torch.exp(-0.5 * ((dof_pos[:, 0] + dof_pos[:, 6]) / morpho_sigma)**2 )
    r_rf_lb_s = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_rf_lb_shoulder))
    kine_diff_rf_lb_knee = torch.exp(-0.5 * ((dof_pos[:, 1] + dof_pos[:, 7]) / morpho_sigma)**2 )
    r_rf_lb_k = 1 - torch.exp(-morpho_kine_coef * (1 - kine_diff_rf_lb_knee))
    E_R_rf_lb = (theta_buffer[:, 1] == theta_buffer[:, 2]) * (r_rf_lb_s + r_rf_lb_k)
    

    judge = (theta_buffer[:, 0] == theta_buffer[:, 2]).float() + (theta_buffer[:, 1] == theta_buffer[:, 3]).float() + (theta_buffer[:, 0] == theta_buffer[:, 1]).float() + (theta_buffer[:, 2] == theta_buffer[:, 3]).float() + (theta_buffer[:, 0] == theta_buffer[:, 3]).float() + (theta_buffer[:, 1] == theta_buffer[:, 2]).float()
    judge_is_zero_indices = torch.where(judge == 0)[0]
    judge[judge_is_zero_indices] = float('inf')
    E_R_morphological_kinematic_coef = rew_scales['E_R_morphological_kinematic_coef'] / judge
    # print(E_R_morphological_kinematic_coef)
    E_R_Morphological = E_R_morphological_kinematic_coef * -(E_R_lf_rf + E_R_lb_rb + E_R_lf_lb + E_R_rf_rb + E_R_lf_rb + E_R_rf_lb)

    # print('f_lr', (theta_buffer[:, 0] == theta_buffer[:, 2]))
    # print('b_lr', (theta_buffer[:, 1] == theta_buffer[:, 3]))
    # print('l_fb', (theta_buffer[:, 0] == theta_buffer[:, 1]))
    # print('r_fb', (theta_buffer[:, 2] == theta_buffer[:, 3]))
    # print('E_R_morphological_kinematic_coef',E_R_morphological_kinematic_coef)
    # print((theta_buffer[:, 0] == theta_buffer[:, 2]).float()+ (theta_buffer[:, 1] == theta_buffer[:, 3]).float() + (theta_buffer[:, 0] == theta_buffer[:, 1]).float() + (theta_buffer[:, 2] == theta_buffer[:, 3].float()) )
    # print('E_R_Morphological',E_R_Morphological)

    # # Older version of morphological symmetry
    # # Kinematic symmetries
    # # # front legs left/right symmetry
    # sigma_tmp1 = 0.05

    # kine_diff1_prob = torch.exp(-0.5 * ((dof_pos[:, 2] + dof_pos[:, 6]) / sigma_tmp1)**2 )
    # kine_diff1 = 1 - torch.exp(-rew_scales['bounding_kinematic_diff_coef'] * (1 - kine_diff1_prob))
    # kine_diff2_prob = torch.exp(-0.5 * ((dof_pos[:, 3] + dof_pos[:, 7]) / sigma_tmp1)**2 )
    # kine_diff2 = 1 - torch.exp(-rew_scales['bounding_kinematic_diff_coef'] * (1 - kine_diff2_prob))
    # # back legs left/right symmetry
    # kine_diff3_prob = torch.exp(-0.5 * ((dof_pos[:, 0] + dof_pos[:, 4]) / sigma_tmp1)**2 )
    # kine_diff3 = 1 - torch.exp(-rew_scales['bounding_kinematic_diff_coef'] * (1 - kine_diff3_prob))
    # kine_diff4_prob = torch.exp(-0.5 * ((dof_pos[:, 1] + dof_pos[:, 5]) / sigma_tmp1)**2 )
    # kine_diff4 = 1 - torch.exp(-rew_scales['bounding_kinematic_diff_coef'] * (1 - kine_diff4_prob))
    # E_R_Bounding_Kinematic = -(kine_diff1 + kine_diff2 + kine_diff3 + kine_diff4)

    # # 2024.1.12: Add dynamics into symmetry constraints
    # # Dynamic symmetries
    # # # front legs left/right symmetry
    # conta_diff1 = contact_forces[:, 7, :]  - contact_forces[:, 15, :]
    # dyna_diff1 = 1 - torch.exp(- rew_scales['bounding_dynamic_diff_coef'] * torch.norm(conta_diff1, dim = -1))
    # # back legs left/right symmetry
    # conta_diff2 = contact_forces[:, 4, :]  - contact_forces[:, 12, :]
    # dyna_diff2 = 1 - torch.exp(- rew_scales['bounding_dynamic_diff_coef'] * torch.norm(conta_diff2, dim = -1))
    # E_R_Bounding_Dynamics = -(dyna_diff1 + dyna_diff2)

    # # 2024.1.12: Add dynamics into symmetry constraints
    # # E_R_Bounding = 0.50 * E_R_Bounding_Kinematic + 0.05 * warmup_coef2 * E_R_Bounding_Dynamics
    # E_R_Bounding = rew_scales['E_R_bounding_kinematic_coef'] * E_R_Bounding_Kinematic + rew_scales['E_R_bounding_dynamic_coef'] * E_R_Bounding_Dynamics

    '''
        2024.02.21: Add a reward term for pitching velocity (now only test on bounding gait).
        4 cases in total:
            F - swing, B - swing: no penalty
            F - swing, B - stance: penalize positive pitching velocity
            F - stance, B - swing: penalize negative pitching velocity
            F - stance, B - stance: no penalty
        Term plugged into the exponential reward function: max(-(C_F - C_B) * pitch_vel, 0), where C means whether the foot is swing (C = -E_C_frc)
    '''
    curr_pitching_ang_vel = base_ang_vel[:, 0]

    last_base_quat = last_root_states[:, 3:7]
    last_pitching_ang_vel = quat_rotate_inverse(last_base_quat, last_root_states[:, 10:13])[:, 0]

    pitching_acc = (curr_pitching_ang_vel - last_pitching_ang_vel) / dt
    # print(pitching_acc)
    pitching_rew_coef = (E_C_frc_lf_val + E_C_frc_rf_val - E_C_frc_lb_val - E_C_frc_rb_val)/2
    pitching_rew_term = torch.clamp(pitching_rew_coef * pitching_acc, max = 0.0)
    pitching_reward = -1 * (1 - torch.exp(rew_scales['pitching_rew_coef'] * pitching_rew_term))
    E_R_pitching = rew_scales['E_R_pitching_coef'] * pitching_reward


    # Compute toral rewards
    total_reward = up_reward + E_R_cmd + E_R_smooth + E_R_Prd + E_R_Morphological + E_R_pitching
    # print('E_R_cmd', E_R_cmd)
    # print('E_R_smooth', E_R_smooth)
    # print('E_R_Prd', E_R_Prd)
    # print('E_R_Bounding', E_R_Bounding)
    total_reward = torch.clip(total_reward, 0, None)

    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.5
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1, dim=1)
    reset = torch.where(up_proj < 0.85, torch.ones_like(reset), reset)

    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset



# 2023.11.08: Modify this function because the previous one is not actually joint state information.
@torch.jit.script
def compute_anymal_observations(root_states: Tensor,
                                # commands: Tensor,
                                dof_pos: Tensor,
                                default_dof_pos: Tensor,
                                dof_vel: Tensor,
                                gravity_vec: Tensor,
                                actions: Tensor,
                                episode_lengths : Tensor,
                                rigid_body_state : Tensor,
                                # 2024.01.29: Add commands
                                commands_linvel_x : Tensor,
                                commands_linvel_y : Tensor,
                                commands_angvel_yaw : Tensor,
                                # 2024.01.29: Add period and duty factors
                                period_buffer : Tensor,
                                max_episode_length : Tensor,
                                num_steps_one_period_buffer : Tensor,
                                duty_factor_buffer : Tensor,
                                # 2024.02.12: Theta buffer
                                theta_buffer : Tensor,
                                # scales
                                lin_vel_scale: float,
                                ang_vel_scale: float,
                                dof_pos_scale: float,
                                dof_vel_scale: float,
                                # theta_left_front : float,
                                # theta_left_rear : float,
                                # theta_right_front : float,
                                # theta_right_rear : float,
                                ) -> Tensor:
    
    # Base height: 1
    base_height = root_states[:, 2].unsqueeze(-1)

    base_quat = root_states[:, 3:7]
    # Base liner velocity in world coordinate: 3
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale 
    # print('base_lin_vel', base_lin_vel)
    # Base angular velocity in world coordinate : 3
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    
    # Joint positions: 8
    joint_pos = (dof_pos - default_dof_pos) * dof_pos_scale
    # Joint velocities: 8
    joint_vel = dof_vel * dof_vel_scale

    # Phase ratios: 2
    r_swing = torch.ones_like(episode_lengths) * (1.0 - duty_factor_buffer)
    r_stance = torch.ones_like(episode_lengths) * duty_factor_buffer
    r = torch.stack([r_swing, r_stance], dim = -1)

    # Clock inputs: 4
    phi = episode_lengths / num_steps_one_period_buffer
    # p = torch.stack([
    #     torch.sin(2 * torch.pi * (phi + theta_left_front)),
    #     torch.sin(2 * torch.pi * (phi + theta_left_rear)),
    #     torch.sin(2 * torch.pi * (phi + theta_right_front)),
    #     torch.sin(2 * torch.pi * (phi + theta_right_rear)),
    # ], dim = -1)
    p = torch.sin(2 * torch.pi * (phi.unsqueeze(-1) + theta_buffer))

    # Velocity command: 3
    target_linvel_x = commands_linvel_x * lin_vel_scale
    target_linvel_y = commands_linvel_y * lin_vel_scale
    target_angvel_yaw = commands_angvel_yaw * ang_vel_scale
    commands = torch.stack([target_linvel_x, target_linvel_y, target_angvel_yaw], -1)

    # 1 + 3 + 3 + 8 + 8 + 2 + 4 + 3 = 32
    obs = torch.cat([
        base_height, 
        base_lin_vel, 
        base_ang_vel, 
        joint_pos, 
        joint_vel,
        r,
        p,
        commands
    ], dim = -1)


    return obs




'''
    Mirroring functions for learning policies that provide symmetric predictions.
'''
# Left/Right 
def obs_LR_mirroring_func(obs):
    mirror_inds = [
        0, # base height
        1, 2, 3, # base linear velocity
        4, 5, 6, # base angular velocity
        11, 12, 13, 14, # right foot joints
        7, 8, 9, 10, # left foot joints,
        19, 20, 21, 22, # right foot joints
        15, 16, 17, 18, # left foot joints
        23, 24, # phase ratios,
        27, 28, # right clock inputs,
        25, 26 # left clock inputs
    ]
    return obs[..., mirror_inds]


def act_LR_mirroring_func(act):
    mirror_inds = [
        4, 5, 6, 7, # right foot joints
        0, 1, 2, 3 # left foot joints
    ]
    return act[..., mirror_inds]





'''
    2023.12.03: Define periodical reward functions 
'''
from scipy.stats import vonmises_line


def limit_input_vonmise_cdf(x, loc, kappa):
    # x: [0, 1]
    return vonmises_line.cdf(x = 2*np.pi*x, loc = 2*np.pi*loc, kappa = kappa)


def P_I(r, start, end, kappa, shift = 0):
    # P(I = 1)
    phi = (r + shift) % 1.0
    P1 = limit_input_vonmise_cdf(phi, start, kappa)
    P2 = limit_input_vonmise_cdf(phi, end, kappa)
    P3 = limit_input_vonmise_cdf(phi, start - 1.0, kappa)
    P4 = limit_input_vonmise_cdf(phi, end - 1.0, kappa)
    P5 = limit_input_vonmise_cdf(phi, start + 1.0, kappa)
    P6 = limit_input_vonmise_cdf(phi, end + 1.0, kappa)
    return P1 * (1 - P2) + P3 * (1 - P4) + P5 * (1 - P6)


def E_I(r, start, end, kappa, shift = 0):
    return 1 * P_I(r, start, end, kappa, shift)


def E_I_swing_frc(r, duty_factor, kappa, shift = 0):
    return E_I(r, 0, 1.0 - duty_factor, kappa, shift)


def E_I_stance_frc(r, duty_factor, kappa, shift = 0):
    return E_I(r, 1.0 - duty_factor, 1.0, kappa, shift)


def E_I_swing_spd(r, duty_factor, kappa, shift = 0):
    return E_I(r, 0, 1.0 - duty_factor, kappa, shift)


def E_I_stance_spd(r, duty_factor, kappa, shift = 0):
    return E_I(r, 1.0 - duty_factor, 1.0, kappa, shift)


def E_C_frc(
    r, 
    duty_factor,
    kappa,
    c_swing_frc,
    c_stance_frc,
    shift = 0
):
    return c_swing_frc * E_I_swing_frc(r, duty_factor, kappa, shift) + c_stance_frc * E_I_stance_frc(r, duty_factor, kappa, shift)


def E_C_spd(
    r, 
    duty_factor,
    kappa,
    c_swing_spd,
    c_stance_spd,
    shift = 0
):
    return c_swing_spd * E_I_swing_spd(r, duty_factor, kappa, shift) + c_stance_spd * E_I_stance_spd(r, duty_factor, kappa, shift)