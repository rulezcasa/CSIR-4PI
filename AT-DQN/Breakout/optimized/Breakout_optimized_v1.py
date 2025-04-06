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
# import matplotlib.pyplot as plt
import cv2
DEBUG = 0

# TODO: Soft update with small tau - Further tuning.
# TODO: Take care of frame squeezing in pong and other games. Optimize specifically.
# TODO: Huber loss instead of MSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

if DEBUG:
    import wandb
    from wandb import AlertLevel


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

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[34:193, :]
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

class QNetwork(nn.Module):
    def __init__(self, state_shape, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity=1000000, device=device):
        self.capacity = capacity
        self.device = device
        self.device_cpu = 'cpu'
        self.position = 0
        self.size = 0
        
        self.states = torch.zeros((capacity, 4, 84, 84), dtype=torch.float32, device=self.device_cpu)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=self.device_cpu)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device_cpu)
        self.next_states = torch.zeros((capacity, 4, 84, 84), dtype=torch.float32, device=self.device_cpu)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device_cpu)
        
        # optimization - pinned memory for faster transfers
        self.states = self.states.pin_memory()
        self.actions = self.actions.pin_memory()
        self.rewards = self.rewards.pin_memory()
        self.next_states = self.next_states.pin_memory() 
        self.dones = self.dones.pin_memory()

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # single operation
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device)
        )


class ATDQNAgent:
    def __init__(
        self,
        action_size,
        state_shape,
        tau=0.45,
        beta_start=0.4,
        beta_end=1.0,
        T=20000000,
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
        self.replay_buffer = ReplayBuffer(capacity=1000000, device=device)
        self.gamma = 0.99
        self.alpha = defaultdict(lambda: torch.tensor(0.25, device=self.device, dtype=torch.float32))
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

    # sha-256 hashing for efficient memory utilisation
    def get_state_hash(self, state):
        return xxhash.xxh3_128(state.tobytes()).hexdigest()

    # used to get attentin weight a state using hash key
    def get_attention(self, state):
        return self.alpha[self.get_state_hash(state)]

    # Min-max normalizing: [0,1]
    def normalize_attention(self):
        if not self.alpha:
            return

        values = torch.stack(list(self.alpha.values()))
        min_val, max_val = values.min(), values.max()

        if min_val == max_val:
            return  # Skip normalization if all values are the same

         # Normalize values on GPU
        normalized_values = (values - min_val) / (max_val - min_val)

        for i, (key, _) in enumerate(self.alpha.items()):
            self.alpha[key] = normalized_values[i]

    # update the attention of a state (importance sampled normalized weight)
    def update_attention(self, state_hashes, IS_td_errors):
        for state_hash, error in zip(state_hashes, IS_td_errors):
            self.attention_history[state_hash].append(error.item())

        for state_hash in state_hashes:
            history_array = np.array(self.attention_history[state_hash], dtype=np.float32)
            attention_value = np.mean(history_array)  # Efficient SMA calculation
            self.alpha[state_hash] = torch.tensor(attention_value, dtype=torch.float32, device=self.device)

        self.normalize_attention()

    # Importance weight computation
    def compute_importance_weight(self, td_errors):
        priority = td_errors.abs() + 1e-6  # Ensure nonzero denominator
        return (1 / priority) ** self.beta  # Vectorized operation

    def act(self, state):
        if self.get_attention(state).item() < self.tau:
            self.exploration_count += 1  # Exploration
            return np.random.choice(self.action_size)
        self.exploitation_count += 1  # Exploitation
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def train_step(self):
        if self.replay_buffer.size < 50000:
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
        importance_weights = self.compute_importance_weight(td_errors.detach())

        state_hashes = [self.get_state_hash(state.cpu().numpy()) for state in states]
        self.update_attention(state_hashes, importance_weights.squeeze())

        # MSE loss
        loss = torch.mean(td_errors**2)

        self.optimizer.zero_grad()
        loss.backward()
        # add grad clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % 10000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item(), td_errors.abs().squeeze().cpu().tolist(), q_values.squeeze().cpu().tolist()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def anneal_beta(self):
        if self.replay_buffer.size >= 50000:
            self.beta = min(self.beta + self.delta_beta, self.beta_end)


def train_agent(env_name, total_steps=20000000, render=False):
    env = gym.make(env_name)
    state, _ = env.reset()
    state = preprocess_frame(state)
    state_stack = np.stack([state] * 4, axis=0)
    state_shape = (4, 84, 84)
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
        action = agent.act(state_stack)
        next_frame, reward, done, _, _ = env.step(action)
        next_frame = preprocess_frame(next_frame)
        next_state_stack = np.concatenate((state_stack[1:], np.expand_dims(next_frame, axis=0)), axis=0)
        agent.add_experience(state_stack, action, reward, next_state_stack, done)
        result = agent.train_step()
        state_stack = next_state_stack
        total_reward += reward
        if result is not None:
            loss, td_error, qvalue = result
            losses.append(loss)
            td_errors_per_episode.extend(td_error)
            q_values.extend(qvalue)

        if done:
            episode += 1
            mean_losses = np.mean(losses) if losses else 0.0
            mean_td_error = np.mean(td_errors_per_episode) if td_errors_per_episode else 0.0
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
                print(f"Episode {episode} - Steps: {step+1}, Reward: {total_reward:.2f}, Loss: {mean_losses:.4f}, "
                      f"Beta: {agent.beta:.4f}, Length: {episode_length}, TD Error: {mean_td_error:.4f}, "
                      f"Q Value: {mean_q_value:.4f}")
            
            if episode % 10 == 0:
                if agent.alpha:  # Ensure there are values to log
                    attention_values = torch.stack(list(agent.alpha.values()))  # Keep on GPU

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
                        if episode % 50 == 0: # increased
                            print(f"Episode {episode} - Attention stats: Mean={mean_val:.4f}, Std={std_val:.4f}, "
                                  f"Min={min_val:.4f}, Max={max_val:.4f}")

            if episode % 100 == 0:
                print(f"Episode {episode}: Mean α = {torch.mean(torch.stack(list(agent.alpha.values()))).item():.4f}, τ = {agent.tau:.4f}")
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
                    print(f"Episode {episode} - Explored: {agent.exploration_count}, Exploited: {agent.exploitation_count}")
                    
                agent.exploration_count = 0
                agent.exploitation_count = 0

            state, _ = env.reset()
            state = preprocess_frame(state)
            state_stack = np.stack([state] * 4, axis=0)
            total_reward = 0
            losses = []
            episode_length = 0
            td_errors_per_episode = []
            q_values = []

        agent.anneal_beta()

    print("Training complete!")
    env.close()

    os.makedirs("AT_DQN_Models", exist_ok=True)
    torch.save(agent.q_network.state_dict(), f"AT_DQN_Models/atdqn_{env_name.split('/')[-1]}_model.pth")
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
                "total_steps": 20000000,
                "beta_start": 0.4,
                "beta_end": 1.0,
                "lr": 0.00025,
                "tau": 0.45,
            },
        )
    else:
        print("Running in non-debug mode, wandb logging disabled")
        print("Config: total_steps=20000000, beta_start=0.4, beta_end=1.0, lr=0.00025, tau=0.45")
        
    # Train agent
    train_agent("ALE/Pong-v5", total_steps=20000000)