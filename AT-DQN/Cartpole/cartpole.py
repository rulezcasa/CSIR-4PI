import numpy as np
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import psutil
import hashlib
import xxhash
import torch.profiler
from tqdm import tqdm
from datetime import timedelta
import cv2
import yaml

DEBUG = True
if DEBUG:
    import wandb
    from wandb import AlertLevel


# TODO: Soft update with small tau - Further tuning.
# TODO: Take care of frame resizing/squeezing in pong and other games. Optimize specifically.
# TODO: Huber loss instead of MSE
# DIDNT HELP: Mixed Precision Training
# DIDNT HELP: Replacing hash with finger printing
# DIDNT HELP: Partial JIT compilation of frame processing, batching


# Initilizations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['device']=='mps':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if config['device']=='cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True



# function to log metrics/system usage
def alert_usage():
    # CPU usage in percentage
    cpu_usage = psutil.cpu_percent(interval=1)

    # RAM usage in percentage
    ram_usage = psutil.virtual_memory().percent

    if ram_usage > 90:
        if DEBUG:
            wandb.alert(
                title="High memory usage",
                text=f"CPU: {cpu_usage:.2f}%",
                level=AlertLevel.WARN,
                wait_duration=timedelta(minutes=10),
            )
        else:
            print(f"ALERT: High memory usage - CPU: {cpu_usage:.2f}%")


class QNetwork(nn.Module):
    def __init__(self, state_shape, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, 32)  # cartpole input : (4,)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(
        self, capacity=config["replay_buffer"], device=device
    ):  # adjusting replay buffer to 50K. ()
        self.capacity = capacity
        self.device = device
        self.device_cpu = "cpu"
        self.position = 0  # used to track the current index in the buffer.
        self.size = 0  # used to track the moving size of the buffer.

        self.states = torch.zeros(
            (capacity, 4), dtype=torch.float32, device=self.device_cpu
        )  # dimension change from 4,84,84 to 4 (cartpole)
        self.actions = torch.zeros(
            (capacity, 1), dtype=torch.long, device=self.device_cpu
        )
        self.rewards = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=self.device_cpu
        )
        self.next_states = torch.zeros(
            (capacity, 4), dtype=torch.float32, device=self.device_cpu
        )  # dimension change
        self.dones = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=self.device_cpu
        )

        optimization - pinned memory for faster transfers
        self.states = self.states.pin_memory()
        self.actions = self.actions.pin_memory()
        self.rewards = self.rewards.pin_memory()
        self.next_states = self.next_states.pin_memory()
        self.dones = self.dones.pin_memory()

    def add(
        self, state, action, reward, next_state, done
    ):  # add experince to the current position of buffer
        self.states[self.position] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.position] = done

        self.position = (
            self.position + 1
        ) % self.capacity  # increment position index (circular buffer)
        self.size = min(self.size + 1, self.capacity)  # increment size

    def sample(self, batch_size):
        indices = np.random.choice(
            self.size, batch_size, replace=False
        )  # sample experiences randomly

        # single operation
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
        )


# optimization - 200hrs
class StateAttentionTrackerLRU:
    def __init__(
        self, capacity=config["LRU"], device=device, history_length=config["sma_window"]
    ):  # sma window from 1000 to 10
        self.device = device
        self.capacity = capacity
        self.history_length = history_length
        self.attention_values = (
            torch.ones(capacity, dtype=torch.float32, device=device) * 0.25
        )
        self.attention_history = torch.zeros(
            (capacity, history_length), dtype=torch.float32, device=device
        )  ## CHECK!
        self.history_counts = torch.zeros(
            capacity, dtype=torch.long, device=device
        )  ## CHECK!

        self.current_index = 0
        self.hash_to_index = {}
        self.last_access = torch.zeros(capacity, dtype=torch.long, device=device)
        self.access_counter = 0  # Timestep for each state accessed

    def get_state_hash(self, state):  # Function to hash state
        if isinstance(state, torch.Tensor):  # if tensor then
            if state.device != torch.device("cpu"):
                state_bytes = state.cpu().numpy().tobytes()
            else:
                state_bytes = state.numpy().tobytes()
        else:  # if array then
            state_bytes = np.asarray(state).tobytes()
        return xxhash.xxh3_64(state_bytes).hexdigest()

    def get_state_index(self, state):  # Retrieve index or create new in LRU
        state_hash = self.get_state_hash(state)
        self.access_counter += 1

        if state_hash in self.hash_to_index:  # state already exists
            idx = self.hash_to_index[state_hash]
            self.last_access[idx] = (
                self.access_counter
            )  # update access time of this state
            return idx  # return the index

        if self.current_index < self.capacity:
            idx = self.current_index
            self.current_index += 1
        else:
            used_indices = torch.arange(self.current_index, device=self.device)
            idx = used_indices[
                torch.argmin(self.last_access[: self.current_index])
            ].item()
            old_hash = None
            for h, i in list(self.hash_to_index.items()):
                if i == idx:
                    old_hash = h
                    break
            if old_hash:
                del self.hash_to_index[old_hash]
            self.history_counts[idx] = 0
            self.attention_history[idx].zero_()

        self.hash_to_index[state_hash] = idx
        self.last_access[idx] = self.access_counter

        return idx  # returns index after removing and adding new

    def batch_get_indices(self, states):  # get index for the batch
        return torch.tensor(
            [self.get_state_index(state) for state in states],
            device=self.device,
            dtype=torch.long,
        )

    def get_attention(self, state):  # get attention value for the state index
        idx = self.get_state_index(state)
        return self.attention_values[idx]

    def batch_get_attention(self, states):  # batch processing
        indices = self.batch_get_indices(states)
        return self.attention_values[indices]

    def update_attention(self, states, importance_weights):
        # Convert to tensor if needed - validate
        if not isinstance(importance_weights, torch.Tensor):
            importance_weights = torch.tensor(
                importance_weights, dtype=torch.float32, device=self.device
            )
        indices = self.batch_get_indices(states)

        # optimization - vectorization
        for i, idx in enumerate(indices):
            pos = self.history_counts[idx] % self.history_length
            self.attention_history[idx, pos] = importance_weights[i]
            self.history_counts[idx] += 1

        unique_indices = torch.unique(indices)
        for idx in unique_indices:
            count = min(self.history_counts[idx].item(), self.history_length)
            if count > 0:
                valid_history = self.attention_history[idx, :count]
                self.attention_values[idx] = valid_history.mean()

        self.normalize_attention()

    def normalize_attention(self):
        if self.current_index == 0:
            return
        used_values = self.attention_values[: self.current_index]
        min_val = used_values.min()
        max_val = used_values.max()

        if min_val == max_val:
            return
        used_values.sub_(min_val).div_(max_val - min_val)

    def compute_importance_weight(self, td_errors, beta):
        priority = td_errors.abs() + 1e-6
        return (1 / priority) ** beta

    def to(self, device):
        self.device = device
        self.attention_values = self.attention_values.to(device)
        self.attention_history = self.attention_history.to(device)
        self.history_counts = self.history_counts.to(device)
        self.last_access = self.last_access.to(device)
        return self


class ATDQNAgent:
    def __init__(
        self,
        action_size,
        state_shape,
        tau=config["tau"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        T=config["T"],
        device=device,
    ):
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_shape, action_size).to(device)
        self.target_network = QNetwork(state_shape, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # gpu optimization - 900hrs
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        self.replay_buffer = ReplayBuffer(capacity=50000, device=device)
        self.attention_tracker = StateAttentionTrackerLRU(
            capacity=100000, device=device
        )
        self.gamma = 0.99
        self.alpha = defaultdict(
            lambda: torch.tensor(0.25, device=self.device, dtype=torch.float32)
        )
        self.attention_history = defaultdict(
            lambda: deque(maxlen=1000)
        )  # Track last 1000 attention values
        self.td_errors = defaultdict(float)
        self.tau = tau

        self.beta = beta_start
        self.beta_end = beta_end
        self.delta_beta = (beta_end - beta_start) / T  # Beta annealing per step
        self.step_count = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.check_replay_size = 5000

    def act(self, state):
        if self.attention_tracker.get_attention(state).item() < self.tau:
            self.exploration_count += 1  # Exploration
            return np.random.choice(self.action_size)
        self.exploitation_count += 1  # Exploitation
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def train_step(self):
        if self.replay_buffer.size < self.check_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(32)

        # optimization - gpu
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            max_next_q = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states).gather(1, actions.long())

        td_errors = targets - q_values

        # Update attention based on TD errors and importance sampling
        importance_weights = self.attention_tracker.compute_importance_weight(
            td_errors.detach(), self.beta
        )
        self.attention_tracker.update_attention(states, importance_weights.squeeze())

        # MSE loss
        loss = torch.mean(td_errors**2)

        self.optimizer.zero_grad()
        loss.backward()
        # add grad clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % config["target_update"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return (
            loss.item(),
            td_errors.abs().squeeze().cpu().tolist(),
            q_values.squeeze().cpu().tolist(),
        )

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def anneal_beta(self):
        if self.replay_buffer.size >= self.check_replay_size:
            self.beta = min(self.beta + self.delta_beta, self.beta_end)


def train_agent(env_name, total_steps=100000, render=False):
    env = gym.make(env_name)
    state, _ = env.reset()
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = ATDQNAgent(action_size, state_shape, device=device)

    total_reward = 0
    losses = []
    episode = 0
    episode_length = 0
    td_errors_per_episode = []
    q_values = []

    for step in tqdm(range(total_steps), desc="Training Progress"):
        episode_length += 1
        action = agent.act(state)
        next_frame, reward, done, _, _ = env.step(action)
        agent.add_experience(state, action, reward, next_frame, done)
        result = agent.train_step()
        state = next_frame
        total_reward += reward
        if result is not None:
            loss, td_error, qvalue = result
            losses.append(loss)
            td_errors_per_episode.extend(td_error)
            q_values.extend(qvalue)

        if done:
            episode += 1
            mean_losses = np.mean(losses) if losses else 0.0
            mean_td_error = (
                np.mean(td_errors_per_episode) if td_errors_per_episode else 0.0
            )
            mean_q_value = np.mean(q_values) if q_values else 0.0

            if DEBUG:
                wandb.log(
                    {
                        "global_step": step + 1,
                        "reward": total_reward,
                        "loss": mean_losses,
                        "beta": agent.beta,
                        "episode_length": episode_length,
                        "mean_td_error": mean_td_error,
                        "mean_q_value": mean_q_value,
                    },
                    step=episode,
                )
            else:
                print(
                    f"Episode {episode} - Steps: {step+1}, Reward: {total_reward:.2f}, Loss: {mean_losses:.4f}, "
                    f"Beta: {agent.beta:.4f}, Length: {episode_length}, TD Error: {mean_td_error:.4f}, "
                    f"Q Value: {mean_q_value:.4f}"
                )

            if episode % 10 == 0:
                if agent.attention_tracker.current_index > 0:
                    attention_values = agent.attention_tracker.attention_values[
                        : agent.attention_tracker.current_index
                    ]

                    # Compute statistics efficiently on GPU
                    mean_val = attention_values.mean().item()
                    std_val = attention_values.std().item()
                    min_val = attention_values.min().item()
                    max_val = attention_values.max().item()

                    if DEBUG:
                        wandb.log(
                            {
                                "attention_mean_10": mean_val,
                                "attention_std_10": std_val,
                                "attention_min_10": min_val,
                                "attention_max_10": max_val,
                            },
                            step=episode,
                        )
                    else:
                        if episode % 50 == 0:  # increased
                            print(
                                f"Episode {episode} - Attention stats: Mean={mean_val:.4f}, Std={std_val:.4f}, "
                                f"Min={min_val:.4f}, Max={max_val:.4f}"
                            )

            if episode % 100 == 0:
                # print(f"Episode {episode}: Mean α = {torch.mean(torch.stack(list(agent.alpha.values()))).item():.4f}, τ = {agent.tau:.4f}")
                alert_usage()
                if DEBUG:
                    wandb.log(
                        {
                            "explored_states": agent.exploration_count,
                            "exploited_states": agent.exploitation_count,
                        },
                        step=episode,
                    )
                else:
                    print(
                        f"Episode {episode} - Explored: {agent.exploration_count}, Exploited: {agent.exploitation_count}"
                    )

                agent.exploration_count = 0
                agent.exploitation_count = 0

            state, _ = env.reset()
            total_reward = 0
            losses = []
            episode_length = 0
            td_errors_per_episode = []
            q_values = []

        agent.anneal_beta()

    print("Training complete!")
    env.close()

    os.makedirs("AT_DQN_Models", exist_ok=True)
    torch.save(
        agent.q_network.state_dict(), f"atdqn_{env_name.split('/')[-1]}_model.pth"
    )
    print(f"Model saved successfully at {model_path}")

    if DEBUG:
        wandb.save(model_path)
        wandb.finish()
    else:
        print("Debug mode disabled, skipping wandb model upload")


if __name__ == "__main__":
    if DEBUG:
        wandb.init(
            project="AT-DQN",
            name="Pong_v5",
            config={
                "total_steps": config["T"],
                "beta_start": config["beta_start"],
                "beta_end": config["beta_end"],
                "lr": config["lr"],
                "tau": config["tau"],
            },
        )
    else:
        print("Running in non-debug mode, wandb logging disabled")
        print(
            f"Config: {config['T']}, {config['beta_start']}, beta_end={config['beta_end']}, lr={config['lr']}, tau={config['tau']}"
        )

    # Train agent
    train_agent("CartPole-v1", total_steps=config["T"])
