import functools
import sacred.config
import numpy as np
import tqdm
import torch
import torch.nn.functional as f
import gym

import paper_experiments.experiments.imitation.nv_environments
from paper_experiments.experiments.imitation import environment_adapters
from algorithms.agents.hindsight import vdi
from algorithms import environment
from policies import gaussian
from workflow import util, reporting
import h5py
from generative import rnvp

experiment = sacred.Experiment("vdi, cheetah - collect n, seed x, temporal smoothing")


@experiment.config
def config():
    policy_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    density_learning_rate = 1e-5
    lr_decay_iterations = []
    policy_l2 = 0.
    density_l2 = 1e-5
    burnin = 9000
    density_burnin = 8000

    density_update_rate = 8000
    target_update_step = .005

    discount_factor = 0.995

    exploration_noise = np.log(0.3).item()
    exploration_decay = [(250000, np.log(0.1))]

    replay_size = 1500000
    density_replay_size = 500000
    min_replay_size = 1000
    batch_size = 256
    step_limit = 0.1

    sequence_length = 4
    temporal_smoothing = 0.
    spatial_smoothing = 0.1
    density_factor = 4.

    num_trajs = 1
    skip_steps = 20
    bc_weight = 0.


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self._h1 = torch.nn.Linear(state_dim, 400)
        self._h2 = torch.nn.Linear(400, 300)

        self._mean_out = torch.nn.Linear(300, action_dim)
        self._action_dim = action_dim

    def forward(self, states: torch.Tensor):
        x = f.leaky_relu(self._h1(states))
        x = f.leaky_relu(self._h2(x))
        mean = f.tanh(self._mean_out(x))
        return mean

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim = 400
        self._h1 = torch.nn.Linear(2 * state_dim + action_dim, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)
        torch.nn.init.constant_(self._v_out.bias, -1)
        torch.nn.init.normal_(self._v_out.weight, std=0.01)

    def forward(self, states: torch.Tensor, actions: torch.tensor, target_states: torch.Tensor):
        target_states = target_states - states
        x = f.leaky_relu(self._h1(torch.cat((states, actions, target_states), dim=1)))
        x = f.leaky_relu(self._h2(x))
        return self._v_out(x)/10

def make_env() -> environment.GymEnv:
    return NoVelEnv(gym.make('NVHalfCheetah-v2'))


class DensityModel(torch.nn.Module):
    def __init__(self, device, state_dim, action_dim):
        super().__init__()
        self._model = rnvp.SimpleRealNVP(
             num_bijectors=5, hdim=400, input_dim=state_dim, context_dim=state_dim + action_dim).to(device)

        self._state_dim = state_dim
        self._action_dim = action_dim

    def log_prob(self, true_samples, context):
        return self._model(true_samples, context)

    def forward(self, *, state: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor):
        target_state = target_state - state
        return self.log_prob(target_state, torch.cat([state, action], dim=1)).mean(dim=1)

class StateDensityModel(torch.nn.Module):
    def __init__(self, device, state_dim, action_dim):
        super().__init__()
        self._model = rnvp.SimpleRealNVP(
            num_bijectors=5, hdim=400, input_dim=state_dim, context_dim=0).to(device)

        self._state_dim = state_dim
        self._action_dim = action_dim

    def forward(self, state: torch.Tensor):
        return self._model(state, None).mean(dim=1)



def make_env() -> environment.GymEnv:
    return environment.GymEnv(gym.make('NVHalfCheetah-v2'))


@experiment.capture
def load_demos(num_trajs: int, skip_steps: int):
    demo_path = "paper_experiments/data/nv_cheetah_demos_gail_format_filtered"

    states = []
    actions = []
    h5file = h5py.File(demo_path, 'r')
    for i in range(num_trajs):
        start_index = skip_steps//2  # non-random start so GAIL uses the same demonstration set
        states.append(h5file['obs_B_T_Do'][i, start_index::skip_steps, :][()])
        actions.append(h5file['a_B_T_Da'][i, start_index::skip_steps, :][()])
    return np.concatenate(states), np.concatenate(actions)

@experiment.capture
def random_rollout_bounds(num_rollouts: int, exploration_noise: float):
    env = make_env()
    states = []
    is_terminal = False
    for i in range(num_rollouts):
        unnormalized_init_policy = PolicyNetwork(env.state_dim, env.action_dim)
        unnormalized_init_policy = gaussian.SphericalGaussianPolicy(
                torch.device('cpu'), env.action_dim, unnormalized_init_policy, fixed_noise=exploration_noise)
        states.append(env.reset())
        print(i)
        while not env.needs_reset:
            action = unnormalized_init_policy.sample(states[-1])
            state, _, is_terminal, _ = env.step(action)
            states.append(state)
    return np.min(states, axis=0).astype(np.float32), np.max(states, axis=0).astype(np.float32)


@experiment.automain
def train(density_learning_rate: float, _config: sacred.config.ConfigDict):
    target_dir = "/home/anon/generated_data/algorithms"
    reporting.register_global_reporter(experiment, target_dir)
    device = torch.device('cuda:0')
    demo_states, _ = load_demos()
    demo_min = np.min(demo_states, axis=0)
    demo_max = np.max(demo_states, axis=0)
    random_min, random_max = random_rollout_bounds(10)
    min_states = np.minimum(demo_min, random_min)
    max_states = np.maximum(demo_max, random_max)
    make_normalized_env = functools.partial(environment_adapters.NormalizedEnv, make_env, min_states, max_states)
    eval_env = make_normalized_env()
    np.savetxt(target_dir + "/normalization", [min_states, max_states])
    experiment.add_artifact(target_dir + "/normalization")

    demo_states = eval_env.normalize_state(demo_states)
    demo_states = torch.from_numpy(demo_states).to(device)
    demo_actions = None
    print(demo_states.shape)

    state_dim = demo_states.shape[1]
    action_dim = eval_env.action_dim

    density_model = DensityModel(device, state_dim, action_dim)
    state_density_model = StateDensityModel(device, state_dim, action_dim)
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    params_parser = util.ConfigParser(vdi.VDIParams)
    params = params_parser.parse(_config)

    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)

    agent = vdi.VDI(make_normalized_env, device, density_model, state_density_model, policy, q1, q2, params, demo_states, demo_actions)

    reporting.register_field("eval_return")
    reporting.finalize_fields()
    trange = tqdm.trange(1000000)
    for iteration in trange:
        agent.update()
        reporting.iterate()
        if iteration % 20000 == 0:
            eval_reward = 0
            for i in range(2):
                state = eval_env.reset()
                cumulative_reward = 0
                while not eval_env.needs_reset:
                    action = agent.eval_action(state)
                    state, reward, is_terminal, _ = eval_env.step(action)
                    cumulative_reward += reward
                eval_reward += cumulative_reward/2
            reporting.iter_record("eval_return", eval_reward)

        if iteration % 10000 == 0:
            policy_path = f"{target_dir}/policy_{iteration}"
            with open(policy_path, 'wb') as f:
                torch.save(agent.freeze_policy(torch.device('cpu')), f)
            experiment.add_artifact(policy_path)
            density_model_path = f"{target_dir}/dm_{iteration}"
            with open(density_model_path, 'wb') as f:
                torch.save(density_model, f)
            experiment.add_artifact(density_model_path)

        trange.set_description(f"{iteration} -- " + reporting.get_description(["return", "eval_return", "density_loss", "actor_loss", "td_loss", "env_steps"]))



#@experiment.automain
def _visualize(density_learning_rate: float, _config: sacred.config.ConfigDict):
    demo_states, demo_actions = load_demos()
    min_states = np.min(demo_states, axis=0)
    max_states = np.max(demo_states, axis=0)

    def make_normalized_env():
        return NormalizedEnv(make_env, min_states, max_states)
    eval_env = make_normalized_env()

    agent = torch.load("/home/anon/generated_data/sacred/124/policy_370000")
    for i in range(20):
        state = eval_env.reset()
        cumulative_reward = 0
        while not eval_env.needs_reset:
            action = agent.sample(state)
            state, reward, is_terminal, _ = eval_env.step(action)
            cumulative_reward += reward
            eval_env._gym_env.env.render("human")
        print(cumulative_reward)

