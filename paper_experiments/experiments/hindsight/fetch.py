from typing import Any
import torch
import torch.nn.functional as f
import gym
import gym.spaces.box
import numpy as np
import tqdm
import sacred

from algorithms import environment
from algorithms.agents.hindsight import her_td3
from workflow import reporting


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim = 400
        self._h1 = torch.nn.Linear(state_dim, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._mean_out = torch.nn.Linear(hdim, action_dim)
        torch.nn.init.constant_(self._mean_out.bias, 0.)
        torch.nn.init.normal_(self._mean_out.weight, std=0.01)
        self._action_dim = action_dim

    def forward(self, states: torch.Tensor):
        x = f.leaky_relu(self._h1(states))
        x = f.leaky_relu(self._h2(x))
        return f.tanh(self._mean_out(x))


class NoNormalizer:
    def normalize_state(self, state):
        return state

    def denormalize_goal(self, goal):
        return goal


class SlideNormalizer:
    object_pos_shift = np.array([-0.75, -0.5, -0.45, -0.75, -0.5, -0.45])
    object_pos_mult = np.array([1/1.25, 1/0.5, 1/0.45])
    rotation_mult = np.array([1/np.pi, 1/np.pi, 1/np.pi])
    object_vel_mult = np.array([1/0.08, 1/0.08, 1/0.08])
    gripper_vel_mult = np.array([1/0.025, 1/0.025])
    gripper_state_mult = np.array([1., 1.])
    rotational_vel_mult = np.array([1., 1., 1.])
    observation_mult = np.concatenate([object_pos_mult, object_pos_mult, object_pos_mult, gripper_state_mult, rotation_mult,
                                       object_vel_mult, rotational_vel_mult, object_vel_mult, gripper_vel_mult])
    def normalize_state(self, state):
        state['observation'][:6] += self.object_pos_shift
        state['observation'] *= self.observation_mult
        state['achieved_goal'] += self.object_pos_shift[:3]
        state['achieved_goal'] *= self.object_pos_mult
        state['desired_goal'] += self.object_pos_shift[:3]
        state['desired_goal'] *= self.object_pos_mult
        return state

    def denormalize_goal(self, goal):
        return goal/self.object_pos_mult[None] - self.object_pos_shift[None, :3]


class FetchEnv(environment.GymEnv):
    goal_dim = 3

    def __init__(self, env_name: str, progressive_noise: bool, small_goal: bool, small_goal_size: float=0.005):
        self._env = gym.make(env_name)
        if small_goal:
            print(f"small goal! ({small_goal_size})")
            self._env.unwrapped.distance_threshold = small_goal_size
        if env_name == 'FetchSlide-v1':
            self._normalizer = SlideNormalizer()  # not strictly necessary but slightly improves performance (~0.1) for both methods
        if env_name == 'FetchPush-v1':
            self._normalizer = NoNormalizer()
        self._env.seed(np.random.randint(10000) * 2)
        self._progressive_noise = progressive_noise
        super().__init__(self._env)
        self._obs_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(28,))

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        while True:
            state = super().reset()
            # reset environment if goal already fulfilled as described in HER paper
            if self._env.compute_reward(state['achieved_goal'], state['desired_goal'], None) != 0.:
                state = self._normalizer.normalize_state(state)
                self._state = np.concatenate([state['desired_goal'], state['observation']])
                return self._state

    def replace_goals(self, transition_sequence: her_td3.HerTransitionSequence, goals: torch.Tensor,
                      replacement_probability: float):
        replace_indices = torch.rand(transition_sequence.states.shape[0], 1) < replacement_probability
        replace_indices = transition_sequence.states.new_tensor(replace_indices)
        transition_sequence.states[:, 0, :self.goal_dim] = replace_indices * goals + (1 - replace_indices) * transition_sequence.states[:, 0, :self.goal_dim]
        transition_sequence.next_states[:, 0, :self.goal_dim] = replace_indices * goals + (1 - replace_indices) * transition_sequence.next_states[:, 0, :self.goal_dim]
        transition_sequence.rewards[:, 0] = transition_sequence.rewards.new(self._env.compute_reward(self._normalizer.denormalize_goal(goals.cpu().numpy()), self._normalizer.denormalize_goal(transition_sequence.states[:, 0, :self.goal_dim].cpu().numpy()), None))
        return her_td3.HerTransitionSequence(
            states=transition_sequence.states.detach(),
            actions=transition_sequence.actions.detach(),
            rewards=transition_sequence.rewards.detach(),
            next_states=transition_sequence.next_states.detach(),
            timeout_weight=transition_sequence.timeout_weight.detach(),
            terminal_weight=transition_sequence.terminal_weight.detach(),
            action_log_prob=transition_sequence.action_log_prob.detach(),
            time_left=transition_sequence.time_left.detach(),
            achieved_goal=transition_sequence.achieved_goal.detach()
        )

    def step(self, action):
        if self._progressive_noise:
            state, original_reward, is_terminal, info = super().step(action + (action**2).mean() * np.random.randn(self.action_dim) * np.exp(-1))
        else:
            state, original_reward, is_terminal, info = super().step(action)

        #if original_reward > -1:
            #print(state['achieved_goal'])
            #print(state['desired_goal'])
        state = self._normalizer.normalize_state(state)
        info['achieved_goal'] = state['achieved_goal']
        self._state = np.concatenate([state['desired_goal'], state['observation']])

        return self._state, original_reward, is_terminal, info


def make_env(env_name: str, progressive_noise: bool, small_goal: bool, small_goal_size: float=0.005) -> FetchEnv:
    return FetchEnv(env_name, progressive_noise, small_goal, small_goal_size)


def train_fetch(experiment: sacred.Experiment, agent: Any, eval_env: FetchEnv, progressive_noise: bool, small_goal: bool):
    reporting.register_field("eval_success_rate")
    reporting.register_field("action_norm")
    reporting.finalize_fields()
    if progressive_noise:
        trange = tqdm.trange(2000000)
    elif small_goal:
        trange = tqdm.trange(2000000)
    else:
        trange = tqdm.trange(2000000)
    for iteration in trange:
        if iteration % 10000 == 0:
            action_norms = []
            success_rate = 0
            for i in range(50):
                state = eval_env.reset()
                while not eval_env.needs_reset:
                    action = agent.eval_action(state)
                    action_norms.append(np.linalg.norm(action))
                    state, reward, is_terminal, info = eval_env.step(action)
                    if reward > -1.:
                        success_rate += 1
                        break
            reporting.iter_record("eval_success_rate", success_rate)
            reporting.iter_record("action_norm", np.mean(action_norms).item())

        if iteration % 20000 == 0:
            policy_path = f"/tmp/policy_{iteration}"
            with open(policy_path, 'wb') as f:
                torch.save(agent.freeze_policy(torch.device('cpu')), f)
            experiment.add_artifact(policy_path)

        agent.update()
        reporting.iterate()
        trange.set_description(f"{iteration} -- " + reporting.get_description(["return", "td_loss", "env_steps"]))


def show_fetch():
    progressive_noise = False
    small_goal = False
    env_name = 'FetchPush-v1'
    eval_env = make_env(env_name, progressive_noise, small_goal)

    agent = torch.load('/home/anon/sacred_partition/param_test_1/deterministic_push_td3/1/policy_340000')
    success_rate = 0
    for i in range(200):
        state = eval_env.reset()
        while not eval_env.needs_reset:
            action = agent.sample(state).squeeze()
            state, reward, is_terminal, info = eval_env.step(action)
            #breakpoint()
            if reward > -1.:
                success_rate += 1
                break
            eval_env.env.render("human")
    print(success_rate)


if __name__ == '__main__':
    show_fetch()
