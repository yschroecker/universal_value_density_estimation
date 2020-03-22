import functools
import sacred.config
import tqdm
import torch
import torch.nn.functional as f
import gym
import h5py

import paper_experiments.experiments.imitation.nv_environments
from paper_experiments.experiments.imitation.environment_adapters import *
from algorithms.agents.hindsight import vdi
from generative import rnvp
from algorithms import environment
from workflow import util, reporting
from policies import gaussian


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self._h1 = torch.nn.Linear(state_dim, 400)
        self._h2 = torch.nn.Linear(400, 300)

        self._mean_out = torch.nn.Linear(300, action_dim)
        # torch.nn.init.constant_(self._mean_out.bias, 0.)
        # torch.nn.init.normal_(self._mean_out.weight, std=0.01)
        self._action_dim = action_dim

    def forward(self, states: torch.Tensor):
        x = f.leaky_relu(self._h1(states))
        x = f.leaky_relu(self._h2(x))
        mean = f.tanh(self._mean_out(x))
        return mean


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, use_one_q: bool):
        super().__init__()
        hdim = 400
        if use_one_q:
            self._h1 = torch.nn.Linear(state_dim + action_dim, hdim)
        else:
            self._h1 = torch.nn.Linear(2 * state_dim + action_dim, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)
        torch.nn.init.constant_(self._v_out.bias, 0.)
        torch.nn.init.normal_(self._v_out.weight, std=0.01)
        self._use_one_q = use_one_q

    def forward(self, states: torch.Tensor, actions: torch.tensor, target_states: torch.Tensor):
        # assert self._h1.in_features == 2 * states.shape[1] + actions.shape[1]
        if self._use_one_q:
            x = f.leaky_relu(self._h1(torch.cat((states, actions), dim=1)))
        else:
            target_states = target_states - states
            x = f.leaky_relu(self._h1(torch.cat((states, actions, target_states), dim=1)))
        x = f.leaky_relu(self._h2(x))
        #x = f.leaky_relu(self._h3(x))
        return self._v_out(x)/10#.exp()

experiment = sacred.Experiment("vdi, humanoid, NC, no term. 100/20, lr scheduler, more critic bleed, clamp, just_one_q=false")


def make_env() -> environment.GymEnv:
    #return environment.GymEnv(gym.make('NoComHumanoid-v2'))
    return ActionEnv(gym.make('NoComHumanoid-v2'))


@experiment.config
def config():
    discount_factor = 0.995
    policy_learning_rate = 3e-4
    critic_learning_rate = 1e-4
    density_learning_rate = 2e-5
    target_update_step = .005
    exploration_noise = np.log(0.1).item()
    replay_size = 1500000
    density_replay_size = 500000
    min_replay_size = 1000
    batch_size = 256
    sample_replay_size = 512 * 100
    sample_frequency = 1
    sample_batch_size = 512 * 1
    burnin = 8000
    validation_fraction = 0.
    num_envs = 1
    sequence_length = 4
    policy_l2 = 0.
    burnin_density_update_rate = 5000
    density_update_rate = 50000
    density_update_rate_burnin = 50000
    density_burnin = 5000
    density_l2 = 1e-6
    critic_l2 = 0.
    temporal_smoothing = 0.98
    spatial_smoothing = 0.1
    context_smoothing = 0.0
    bc_weight = 0.
    num_bijectors = 5
    density_factor = 4

    use_one_q = False

    skip_steps = 20
    num_trajs = 1
    step_limit: float = 0.1



class DensityModel(torch.nn.Module):
    def __init__(self, device, state_dim, action_dim, num_bijectors):
        super().__init__()
        self._model = rnvp.SimpleRealNVP(
             num_bijectors=num_bijectors, hdim=400, input_dim=state_dim, context_dim=state_dim + action_dim).to(device)

        self._state_dim = state_dim
        self._action_dim = action_dim

    def log_probx(self, true_samples, context):
        return self._model(true_samples, context)

    def forward(self, *, state: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor):
        target_state = target_state - state
        return self.log_probx(target_state, torch.cat([state, action], dim=1)).mean(dim=1)# + np.log(10)

class StateDensityModel(torch.nn.Module):
    def __init__(self, device, state_dim, action_dim, num_bijectors):
        super().__init__()
        self._model = rnvp.SimpleRealNVP(
            num_bijectors=num_bijectors, hdim=400, input_dim=state_dim, context_dim=0).to(device)

        self._state_dim = state_dim
        self._action_dim = action_dim

    def forward(self, state: torch.Tensor):
        return self._model(state, None).mean(dim=1)


@experiment.capture
def load_demos(skip_steps: int, num_trajs: int):
    demo_path = "paper_experiments/data/nocom_humanoid_demos_gail_format" # 100 full trajs

    states = []
    actions = []
    h5file = h5py.File(demo_path, 'r')
    random_state = np.random.RandomState(0) # so that visualization has the same normalization
    for i in range(num_trajs):
        start_index = skip_steps//2
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

@experiment.capture
def load_demos_action_env(skip_steps: int, num_trajs: int):
    demo_path = "paper_experiments/data/nocom_humanoid_demos_gail_format" # 100 full trajs

    states = []
    actions = []
    h5file = h5py.File(demo_path, 'r')
    random_state = np.random.RandomState(0) # so that visualization has the same normalization
    for i in range(num_trajs):
        start_index = skip_steps//2
        base_states = h5file['obs_B_T_Do'][i, start_index+1::skip_steps, :][()]
        base_actions = h5file['a_B_T_Da'][i, start_index:-1:skip_steps, :][()]
        states.append(np.concatenate([base_states, base_actions], axis=1))
    return np.concatenate(states)



@experiment.automain
def train(density_learning_rate: float, num_bijectors: int, use_one_q: bool, _config: sacred.config.ConfigDict):
    target_dir = "/home/anon/generated_data/algorithms"
    device = torch.device('cuda:0')
    reporting.register_global_reporter(experiment, target_dir)

    #demo_states, demo_actions = load_demos()
    demo_states = load_demos_action_env()
    demo_min = np.min(demo_states, axis=0)
    demo_max = np.max(demo_states, axis=0)
    random_min, random_max = random_rollout_bounds(10)

    min_states = np.minimum(demo_min, random_min)
    max_states = np.maximum(demo_max, random_max)

    np.savetxt(target_dir + "/normalization", [min_states, max_states])
    experiment.add_artifact(target_dir + "/normalization")
    make_normalized_env = functools.partial(NormalizedEnv, make_env, min_states, max_states)
    eval_env = make_normalized_env()

    demo_states = eval_env.normalize_state(demo_states)

    demo_states = torch.from_numpy(demo_states).to(device)
    print(demo_states.shape)
    demo_actions = None

    state_dim = demo_states.shape[1]
    action_dim = eval_env.action_dim

    density_model = DensityModel(device, state_dim, action_dim, num_bijectors)
    state_density_model = StateDensityModel(device, state_dim, action_dim, num_bijectors)
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    params_parser = util.ConfigParser(vdi.VDIParams)
    params = params_parser.parse(_config)

    q1 = QNetwork(state_dim, action_dim, use_one_q).to(device)
    q2 = QNetwork(state_dim, action_dim, use_one_q).to(device)

    agent = vdi.VDI(make_normalized_env, device, density_model, state_density_model, policy, q1, q2, params,
                    demo_states, demo_actions=demo_actions)

    reporting.register_field("eval_return")
    reporting.finalize_fields()
    trange = tqdm.trange(2500000)
    num_rollouts = 4
    for iteration in trange:
        agent.update()
        reporting.iterate()
        if iteration % 20000 == 0:
            eval_reward = 0
            for i in range(num_rollouts):
                state = eval_env.reset()
                cumulative_reward = 0
                while not eval_env.needs_reset:
                    action = agent.eval_action(state)
                    state, reward, is_terminal, _ = eval_env.step(action)
                    cumulative_reward += reward
                    # eval_env.env.render("human")
                eval_reward += cumulative_reward/num_rollouts
            reporting.iter_record("eval_return", eval_reward)

        if iteration % 10000 == 0:
            policy_path = f"{target_dir}/policy_{iteration}"
            with open(policy_path, 'wb') as f:
                torch.save(agent.freeze_policy(torch.device('cpu')), f)
            experiment.add_artifact(policy_path)
            q_path = f"{target_dir}/q_{iteration}"
            with open(q_path, 'wb') as f:
                torch.save(q1, f)
            experiment.add_artifact(q_path)
            state_density_model_path = f"{target_dir}/sdm_{iteration}"
            with open(state_density_model_path, 'wb') as f:
                torch.save(state_density_model, f)
            experiment.add_artifact(state_density_model_path)
            density_model_path = f"{target_dir}/dm_{iteration}"
            with open(density_model_path, 'wb') as f:
                torch.save(density_model, f)
            experiment.add_artifact(density_model_path)

        trange.set_description(f"{iteration} -- " + reporting.get_description(["return",  "density_loss", "bc_loss", "actor_loss", "td_loss", "env_steps"]))


@experiment.automain
def _visualize(density_learning_rate: float, _config: sacred.config.ConfigDict):
    xid = "48"
    iteration = "100000"
    min_states, max_states = np.loadtxt(f"/home/anon/generated_data/sacred/{xid}/normalization")
    device = torch.device('cuda:0')

    def make_normalized_env():
        return NormalizedEnv(make_env, min_states, max_states)
    eval_env = make_normalized_env()
    np.set_printoptions(edgeitems=100, suppress=True)
    demo_states = load_demos_action_env()
    demo_states = eval_env.normalize_state(demo_states)
    demo_states = torch.from_numpy(demo_states).float().to(device)[::100]


    agent = torch.load(f"/home/anon/generated_data/sacred/{xid}/policy_{iteration}")
    dm = torch.load(f"/home/anon/generated_data/sacred/{xid}/dm_{iteration}")
    sdm = torch.load(f"/home/anon/generated_data/sacred/{xid}/sdm_{iteration}")
    q_model = torch.load(f"/home/anon/generated_data/sacred/{xid}/q_{iteration}")
    q_model._use_one_q = True
    for i in range(20):
        state = eval_env.reset()
        cumulative_reward = 0
        while not eval_env.needs_reset:
            action = agent.mode_np(state)[0]
            state, reward, is_terminal, info = eval_env.step(action)
            cumulative_reward += reward
            if i > 0:
                state_tensor = torch.from_numpy(state[None, :]).float().to(device)
                action_tensor = torch.from_numpy(action[None, :]).float().to(device)
                dm_values = dm(state=state_tensor.repeat(250, 1), action=action_tensor.repeat(250, 1),
                        target_state=demo_states).exp().cpu().detach().numpy()
                sdm_values = sdm(demo_states).exp().cpu().detach().numpy()
                q_values = q_model(state_tensor, action_tensor, state_tensor)
            eval_env._gym_env.env.render("human")
        print(cumulative_reward)


