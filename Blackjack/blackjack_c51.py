# blackjack_c51.py
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from gymnasium.spaces import Box
from RL.utils.non_atari_utils import plotBlackjackLearning

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    plot_distribution: bool = True
    """whether to plot the reward distribution at given episodic intervals"""
    plot_intervals: int = 100
    """interval length between plots of reward distribution"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Blackjack-v1"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -100
    """the return lower bound"""
    v_max: float = 100
    """the return upper bound"""
    buffer_size: int = 2500
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def flatten_observation(observation):
    return np.array(observation, dtype=np.float32).reshape(-1)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_actions, n_atoms=51, v_min=-100, v_max=100):
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = n_actions
        input_dimension = 3
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits = self.network(x)
        batch_size = x.size(0)
        pmfs = torch.softmax(logits.view(batch_size, self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(batch_size), action]


    def get_all_action_pmfs(self, x):
        """
        Retrieves the PMFs for all actions given a state.

        :param x: The input state(s)
        :return: PMFs for all actions
        """
        logits = self.network(x)
        batch_size = logits.size(0)
        pmfs = torch.softmax(logits.view(batch_size, self.n, self.n_atoms), dim=2)
        return pmfs


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def plot_reward_distribution(model, state, episode_number, distribution_plots_dir):
    """
    Plots the reward distribution for all actions given a state.

    :param model: The QNetwork model
    :param state: The input state for which to plot the distributions. This should already be a tensor.
    """
    pmfs = model.get_all_action_pmfs(state)
    pmfs = pmfs.squeeze(0)
    atoms = model.atoms.cpu().numpy()

    fig, ax = plt.subplots()
    for action in range(model.n):
        pmf = pmfs[action].cpu().detach().numpy()
        ax.plot(atoms, pmf, label=f'Action {action}')

    ax.set_title('Estimated Reward Distribution for All Actions')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Probability')
    ax.legend()

    plot_filename = f"episode_{episode_number}_distribution.png"
    plot_path = os.path.join(distribution_plots_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    completed_episodes, win_count, loss_count, draw_count, score = 0, 0, 0, 0, 0
    scores, total_rewards, wins, losses, draws, eps_history = [], [], [], [], [], []

    distribution_plots_dir = os.path.join("distribution_plots", run_name)
    os.makedirs(distribution_plots_dir, exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device: " + str(device))

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    n_actions = envs.single_action_space.n  # Number of actions in the Blackjack environment

    # Instantiate QNetwork with the correct number of actions and the predefined n_atoms, v_min, and v_max
    q_network = QNetwork(n_atoms=args.n_atoms, n_actions=n_actions, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(n_atoms=args.n_atoms, n_actions=n_actions, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    new_observation_space = Box(low=np.array([0, 1, 0]), high=np.array([31, 10, 1]), dtype=np.float32)
    rb = ReplayBuffer(
        args.buffer_size,
        new_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = flatten_observation(obs)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        eps_history.append(epsilon)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs = torch.Tensor(obs).to(device)  # Assuming obs is your observation
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)  # Add a batch dimension if it's a single observation
            actions, pmf = q_network.get_action(obs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = flatten_observation(next_obs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_reward = info["episode"]["r"]
                    total_rewards.append(episode_reward)
                    score += episode_reward
                    if episode_reward == 1:
                        win_count += 1
                    elif episode_reward == -1:
                        loss_count += 1
                    else:
                        draw_count += 1
                    if len(total_rewards) > 50:
                        total_rewards.pop(0)
                    completed_episodes += 1
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    if completed_episodes % 10 == 0:
                        average_reward = sum(total_rewards) / len(total_rewards)
                        print(
                            f"Average reward of the last 50 episodes after {completed_episodes} episodes: {average_reward}")
                        writer.add_scalar("charts/average_reward_last_50", average_reward, global_step)

                    total_games = win_count + loss_count + draw_count
                    wins.append(win_count / total_games if total_games > 0 else 0)
                    losses.append(loss_count / total_games if total_games > 0 else 0)
                    draws.append(draw_count / total_games if total_games > 0 else 0)
                    scores.append(score)
                    score = 0

                    if args.plot_distribution:
                        if completed_episodes % args.plot_intervals == 0 and completed_episodes > 0:
                            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                            actions, pmf = q_network.get_action(obs_tensor)
                            plot_reward_distribution(q_network, obs_tensor, completed_episodes, distribution_plots_dir)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        obs_cpu = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
        actions_cpu = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions

        rb.add(obs_cpu, real_next_obs, actions_cpu, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.blackjack_c51_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from RL.utils.atari_utils import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    x = [i + 1 for i in range(total_games)]
    folder_path = 'plots/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = folder_path + 'blackjack_c51.png'
    plotBlackjackLearning(x, scores, wins, losses, draws, filename)

    envs.close()
    writer.close()
