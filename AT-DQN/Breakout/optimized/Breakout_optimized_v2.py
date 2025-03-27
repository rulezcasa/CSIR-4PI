import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import wandb
from tqdm import tqdm
import psutil
import hashlib
from wandb import AlertLevel
import xxhash
import torch.profiler


# function to log metrics/system usage
def alert_usage():
    # CPU usage in percentage
    cpu_usage = psutil.cpu_percent(interval=1)

    # RAM usage in percentage
    ram_usage = psutil.virtual_memory().percent

    if ram_usage > 90:
        wandb.alert(
            title="High memory usage",
            text=f"CPU: {cpu_usage:.2f}%",
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=10),
        )


# Preprocessing function
def preprocess_frame(frame):
    frame = torch.tensor(frame, dtype=torch.float32).mean(dim=-1)
    frame = torch.nn.functional.interpolate(
        frame.unsqueeze(0).unsqueeze(0), size=(84, 84)
    ).squeeze()
    return frame


# Define the Q-network using PyTorch
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


# Replay Buffer for Experience Replay (modified to use gpu)
class ReplayBuffer:
    def __init__(self, capacity=1000000, device="cuda"):
        self.buffer = deque(maxlen=capacity)
        self.device=device

    def add(self, experience):
        self.buffer.append(tuple(np.array(x, dtype=np.float32) for x in experience))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)  # Sample on CPU
        batch = [self.buffer[i] for i in indices]

        return tuple(torch.tensor(np.array(x), dtype=torch.float32, device=self.device) for x in zip(*batch))

    def size(self):
        return len(self.buffer)


class ATDQNAgent:
    def __init__(
        self,
        action_size,
        state_shape,
        tau=0.45,
        beta_start=0.2,
        beta_end=0.8,
        T=20000000,
        device="cuda",
    ):
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_shape, action_size).to(device)
        self.target_network = QNetwork(state_shape, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        self.replay_buffer = ReplayBuffer()
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
        if self.replay_buffer.size() < 50000:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(32)

        target_q_values = self.target_network(next_states).detach()
        max_next_q = target_q_values.max(dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states).gather(1, actions.long().unsqueeze(1)).squeeze()
    
        td_errors = targets - q_values

        # Update attention based on TD errors and importance sampling
        importance_weights = self.compute_importance_weight(td_errors)

        state_hashes = [self.get_state_hash(state.cpu().numpy()) for state in states]
        self.update_attention(state_hashes, importance_weights)

        # MSE loss
        loss = torch.mean(td_errors**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % 10000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item(), td_errors.abs().cpu().tolist(), q_values.cpu().tolist()

    def anneal_beta(self):
        if self.replay_buffer.size() >= 50000:
            self.beta = min(self.beta + self.delta_beta, self.beta_end)


wandb.init(
    project="AT-DQN",
    name="Breakout_v1",
    config={
        "total_steps": 20000000,
        "beta_start": 0.2,
        "beta_end": 0.8,
        "lr": 0.00025,
        "tau": 0.45,
    },
)
env = gym.make("ALE/Breakout-v5")
state, _ = env.reset()
state_shape = (4, 84, 84)
action_size = env.action_space.n
agent = ATDQNAgent(
    action_size, state_shape, device="cuda"
)

total_steps = 20000000  # Training for 20 million steps
total_cumulative_reward = 0
state = preprocess_frame(state)
state_stack = np.stack([state] * 4, axis=0)
total_reward = 0
losses = []
episode = 0
episode_length = 0
td_errors_per_episode = []
q_values = []

for step in tqdm(range(total_steps), desc="Training Progress"):
    episode_length += 1  # Tracking episode length in steps
    action = agent.act(state_stack)
    next_frame, reward, done, _, _ = env.step(action)
    next_frame = preprocess_frame(next_frame)
    next_state_stack = np.concatenate(
        (state_stack[1:], np.expand_dims(next_frame, axis=0)), axis=0
    )

    agent.replay_buffer.add((state_stack, action, reward, next_state_stack, done))
    result = agent.train_step()

    state_stack = next_state_stack
    total_reward += reward
    if result is not None:
        loss, td_error, qvalues = result
        losses.append(loss)
        td_errors_per_episode.extend(
            td_error
        )  # Track TD errors per sample for each episode
        q_values.extend(qvalues)  # Track TD errors per sample for each episode

    if done:
        episode += 1
        mean_losses = np.mean(losses) if losses else 0.0
        mean_td_error = np.mean(td_errors_per_episode) if td_errors_per_episode else 0.0
        mean_q_value = np.mean(q_values) if q_values else 0.0

        wandb.log(
            {
                "global stepcount after every episode": step + 1,
                "Reward per episode": total_reward,
                "Mean loss per episode": mean_losses,
                "Beta per episode": agent.beta,
                "Episode length": episode_length,
                "Mean TD Error per episode": mean_td_error,
                "Mean Q value per episode": mean_q_value,
            },
            step=episode,
        )

        # Log every 10 episodes
        if episode % 10 == 0:
            if agent.alpha:  # Ensure there are values to log
                attention_values = torch.stack(list(agent.alpha.values()))  # Keep on GPU

                # Compute statistics efficiently on GPU
                mean_val = attention_values.mean().item()
                std_val = attention_values.std().item()
                min_val = attention_values.min().item()
                max_val = attention_values.max().item()

                # Log to wandb
                wandb.log(
                {
                    "Attention Mean every 10 episodes": mean_val,
                    "Attention Std every 10 episodes": std_val,
                    "Attention Min every 10 episodes": min_val,
                    "Attention Max every 10 episodes": max_val,
                },
                step=episode,
                )

        # Log every 100 episodes
        if episode % 100 == 0:
            # print(f"Episode {episode}: Mean α = {np.mean(list(agent.alpha.values()))}, τ = {agent.tau}")
            alert_usage()
            wandb.log(
                {
                    "Total Explored States every 100 episodes": agent.exploration_count,
                    "Total Exploited States every 100 episodes": agent.exploitation_count,
                },
                step=episode,
            )
            agent.exploration_count = 0
            agent.exploitation_count = 0

        state, _ = env.reset()
        state = preprocess_frame(state)
        state_stack = np.stack(
            [state] * 4, axis=0
        )  # stacks the first frame 4 times discarding previous frames
        total_reward = 0
        losses = []
        episode_length = 0
        td_errors_per_episode = []

    agent.anneal_beta()

print("Training complete!")
env.close()


# Define the path to save the model
model_path = "AT_DQN_Models/atdqn_breakoutv1_model.pth"

# Save the model
torch.save(agent.target_network.state_dict(), model_path)
print(f"✅ Model saved successfully at {model_path}")

# Also log the model to Weights & Biases (wandb)
wandb.save(model_path)

wandb.finish()
