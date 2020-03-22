import functools

from algorithms.agents.hindsight import uvd
from workflow import util
from generative import rnvp
from paper_experiments.experiments.hindsight.fetch import *

experiment = sacred.Experiment("Fetch - UVD")


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
        return self._v_out(x)


# noinspection PyUnresolvedReferences
class DensityEstimator(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, reward_factor: float, num_bijectors: int):
        super().__init__()
        self._reward_factor = reward_factor
        self._model = rnvp.SimpleRealNVP(2, state_dim + action_dim, 300, num_bijectors)

    def forward(self, goal: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        goal = torch.squeeze(goal, dim=1)[:, :2]
        states = torch.squeeze(states, dim=1)
        actions = torch.squeeze(actions, dim=1)
        context = torch.cat([states, actions], dim=1)
        # noinspection PyCallingNonCallable
        goal = goal - states[:, 6:8]
        goal_log_pdf = self._model(goal, context).sum(dim=1)
        return goal_log_pdf

    def reward(self, goal: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        # noinspection PyCallingNonCallable
        return self(goal, states, actions).exp() * self._reward_factor

# noinspection PyUnusedLocal
@experiment.config
def _config():
    # this is scaled for HER and UVD due to different computational speed. env_steps/iteration should be comparable
    num_envs = 1
    reward_factor = 0.1
    num_bijectors = 5
    policy_learning_rate = 2e-4
    critic_learning_rate = 2e-4
    small_goal_size = 0.005

# noinspection DuplicatedCode
@experiment.automain
def _run(env_name: str, progressive_noise: bool, reward_factor: float, small_goal: bool, small_goal_size: float, num_bijectors: int, _config):
    device = torch.device('cuda:0')
    target_dir = "/home/anon/generated_data/algorithms"
    reporting.register_global_reporter(experiment, target_dir)
    eval_env = make_env(env_name, progressive_noise, small_goal, small_goal_size)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    density_model = DensityEstimator(state_dim, action_dim, reward_factor, num_bijectors).to(device)
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    params_parser = util.ConfigParser(uvd.UVDParams)
    params = params_parser.parse(_config)
    agent = uvd.UVDTD3(functools.partial(make_env, env_name, progressive_noise, small_goal), device, density_model, q1, q2,
                       policy, params)
    train_fetch(experiment, agent, eval_env, progressive_noise, small_goal)



