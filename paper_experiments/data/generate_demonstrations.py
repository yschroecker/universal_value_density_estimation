from typing import BinaryIO

import gym
import torch
import torch.nn.functional as f
import numpy as np
import h5py
import tqdm
import argparse

from algorithms import environment
from paper_experiments.experiments.imitation import nv_environments


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self._h1 = torch.nn.Linear(state_dim, 400)
        self._h2 = torch.nn.Linear(400, 300)

        self._mean_out = torch.nn.Linear(300, action_dim)
        torch.nn.init.constant_(self._mean_out.bias, 0.)
        torch.nn.init.normal_(self._mean_out.weight, std=0.01)
        self._action_dim = action_dim

    def forward(self, states: torch.Tensor):
        x = f.leaky_relu(self._h1(states))
        x = f.leaky_relu(self._h2(x))
        mean = f.tanh(self._mean_out(x))
        return mean

def generate_demos(env_name: str, expert_policy_path: BinaryIO, num_demos: int, out_name: str, gail_format: bool,
        min_reward: float=-np.float('inf')):
    expert_policy = torch.load(expert_policy_path)
    no_vel = env_name[:2] == 'nv' #lower case! upper case is a real env
    if no_vel:
        env_name = env_name[2:]
    eval_env = environment.GymEnv(gym.make(env_name))

    if gail_format:
        h5file = h5py.File(f"experimental/imitation/uvd_il/data/{out_name}_gail_format",'w')

        max_len = 1000  # assuming a walker, not reacher
        h5file.create_dataset("a_B_T_Da", (num_demos, max_len, eval_env.action_dim), dtype='f')
        h5file.create_dataset("obs_B_T_Do", (num_demos, max_len, eval_env.state_dim), dtype='f')
        h5file.create_dataset("r_B_T", (num_demos, max_len), dtype='f')
        h5file.create_dataset("len_B", (num_demos,), dtype='i')



    demo_states = []
    demo_actions = []
    i = 0
    while i < num_demos:
        print(i)
        traj_states = []
        traj_actions = []
        traj_rewards = []

        state = eval_env.reset()
        if no_vel:
            state[state.shape[0]//2] = 0
        cumulative_return = 0
        while not eval_env.needs_reset:
            action = expert_policy.mode_np(state)[0]
            next_state, reward, is_terminal, _ = eval_env.step(action)
            if no_vel:
                next_state[state.shape[0]//2] = 0
            cumulative_return += reward
            traj_states.append(state)
            traj_actions.append(action)
            traj_rewards.append(reward)
            state = next_state
        if cumulative_return < min_reward:
            continue
        demo_states.extend(traj_states)
        demo_actions.extend(traj_actions)
        print(cumulative_return)


        if gail_format:
            demo_path = f"experimental/imitation/uvd_il/data/{i}_{out_name}_gail"
            h5file['obs_B_T_Do'][i, :len(traj_states), :] = np.array(traj_states)
            h5file['a_B_T_Da'][i, :len(traj_states), :] = np.array(traj_actions)
            h5file['r_B_T'][i, :len(traj_states)] = np.array(traj_rewards)
            h5file['len_B'][i] = len(traj_states)
        else:
            demo_path = f"experimental/imitation/uvd_il/data/{i}_{out_name}"
            with open(demo_path, 'wb') as f:
                np.savez(f, states=demo_states, actions=demo_actions)
        i += 1

def _run():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('expert_policy_path', type=argparse.FileType('rb'))
    parser.add_argument('num_demos', type=int)
    parser.add_argument('out_name', type=str)
    parser.add_argument('--min_reward', type=float, default=-np.float('inf'))
    parser.add_argument('--gail-format', action='store_true')
    args = parser.parse_args()
    generate_demos(args.env_name, args.expert_policy_path, args.num_demos, args.out_name, args.gail_format,
            args.min_reward)


if __name__ == '__main__':
    _run()

