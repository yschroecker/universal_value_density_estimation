import functools

from workflow import util
from algorithms.agents import td3
from paper_experiments.experiments.hindsight.fetch import *

experiment = sacred.Experiment("Fetch - TD3")


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim = 500
        self._h1 = torch.nn.Linear(state_dim + action_dim, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch.Tensor, actions: torch.tensor):
        x = f.leaky_relu(self._h1(torch.cat((states, actions), dim=1)))
        x = f.leaky_relu(self._h2(x))
        return f.tanh(self._v_out(x)) * 50


# noinspection PyUnusedLocal
@experiment.config
def _config():
    # this is scaled for HER and UVD due to different computational speed. env_steps/iteration should be comparable
    num_envs = 4
    policy_learning_rate = 2e-4
    critic_learning_rate = 2e-4
    small_goal_size = 0.005


# noinspection DuplicatedCode
@experiment.automain
def _run(env_name: str, progressive_noise: bool, small_goal: bool, small_goal_size: float,  _config):
    device = torch.device('cuda:0')
    target_dir = "/home/anon/generated_data/algorithms"
    reporting.register_global_reporter(experiment, target_dir)
    eval_env = make_env(env_name, progressive_noise, small_goal, small_goal_size)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    params_parser = util.ConfigParser(td3.TD3Params)
    params = params_parser.parse(_config)
    agent = td3.TD3(functools.partial(make_env, env_name, progressive_noise, small_goal, small_goal_size), device, q1, q2, policy, params)
    train_fetch(experiment, agent, eval_env, progressive_noise, small_goal)
