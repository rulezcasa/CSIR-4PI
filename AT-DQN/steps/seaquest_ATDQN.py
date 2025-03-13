import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import wandb
from tqdm import tqdm
import logging
import sys
import os	
import psutil
import hashlib


# Preprocessing function
def preprocess_frame(frame):
    frame = torch.tensor(frame, dtype=torch.float32).mean(dim=-1)  
    frame = torch.nn.functional.interpolate(frame.unsqueeze(0).unsqueeze(0), size=(84, 84)).squeeze()
    return frame.numpy().astype(np.uint8)
    
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
        
        	
# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
        
        
class ATDQNAgent:
    def __init__(self, action_size, state_shape, tau=0.5, beta_start=0.4, beta_end=1.0, T=1_000_000, device="mps"):
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_shape, action_size).to(device)
        self.target_network = QNetwork(state_shape, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99

        self.alpha = defaultdict(lambda: 1.0)
        self.td_errors = defaultdict(float)
        self.tau = tau

        self.beta = beta_start
        self.beta_end = beta_end
        self.delta_beta = (beta_end - beta_start) / T  # Beta annealing per step
        self.step_count = 0

    #sha-256 hashing for efficient memory utilisation
    def get_state_hash(self, state):
        return hashlib.sha256(state.tobytes()).hexdigest()

    #used to get attentin weight a state using hash key
    def get_attention(self, state):
        return self.alpha[self.get_state_hash(state)]

    #update the attention of a state (importance sampled normalied weight)
    def update_attention(self, state, IS_td_error):
        self.alpha[state] = IS_td_error

    #Importance weight computation
    def compute_importance_weight(self, td_error, N):
        priority = (abs(td_error) + 1e-6) ** 0.5
        importance_weight = (1 / (N * priority)) ** self.beta
        return importance_weight

    def act(self, state):
        if self.get_attention(state) > self.tau:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def train_step(self):
        if self.replay_buffer.size() < 50000:
            return None
        batch = self.replay_buffer.sample(32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(int)).to(self.device)

        target_q_values = self.target_network(next_states).detach()
        max_next_q = target_q_values.max(dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states).gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze()
        td_errors = targets - q_values

        # Update attention based on TD errors and importance sampling
        N = len(self.replay_buffer.buffer)
        importance_weights = torch.FloatTensor([
            self.compute_importance_weight(td_errors[i], N) for i in range(32)
        ]).to(self.device)
        importance_weights /= importance_weights.max() # normalizing : [0,1]

        # using importance weights to update attention for the batch
        for i in range(32):
            state_hash=self.get_state_hash(states[i].cpu().numpy())
            self.update_attention(state_hash, importance_weights[i].item())

        # MSE loss 
        loss = torch.mean(td_errors ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % 10000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


    def anneal_beta(self):
        self.beta = min(self.beta + self.delta_beta, self.beta_end)


wandb.init(project="AT-DQN", name="Seaquest_steps", config={"total_steps": 10000000})
reward_td_table = wandb.Table(columns=["Reward", "Mean loss"])
env = gym.make('ALE/Seaquest-v5')
state, _ = env.reset()
state_shape = (4, 84, 84)
action_size = env.action_space.n
agent = ATDQNAgent(action_size, state_shape, device="mps" if torch.cuda.is_available() else "cpu")

total_steps = 10000000  # Training for 10 million steps
reward_memory = deque(maxlen=100)
total_cumulative_reward = 0
state = preprocess_frame(state)
state_stack = np.stack([state] * 4, axis=0)
total_reward = 0
losses = []

for step in tqdm(range(total_steps), desc="Training Progress"):
    action = agent.act(state_stack)
    next_frame, reward, done, _, _ = env.step(action)
    next_frame = preprocess_frame(next_frame)
    next_state_stack = np.concatenate((state_stack[1:], np.expand_dims(next_frame, axis=0)), axis=0)

    agent.replay_buffer.add((state_stack, action, reward, next_state_stack, done))
    loss = agent.train_step()

    state_stack = next_state_stack
    total_reward += reward
    if loss is not None:
        losses.append(loss)
    

    if done:
        reward_memory.append(total_reward)
        mean_losses = np.mean(losses) if losses else 0.0
        reward_td_table.add_data(total_reward, mean_losses)

        if step>50000:
            print(f"Steps:{step+1}, Reward:{total_reward}")

        wandb.log({
            "Steps per episode": step + 1,
            "Reward per episode": total_reward,
            "Mean loss per episode": mean_losses,
            "Beta per episode": agent.beta,
            "Reward vs mean loss per episode": reward_td_table
        })
        
        state, _ = env.reset()
        state = preprocess_frame(state)
        state_stack = np.stack([state] * 4, axis=0)
        total_reward = 0
        losses = []

    agent.anneal_beta()

wandb.finish()
env.close()


# Define the path to save the model
model_path = "AT_DQN_Models/atdqn_seaquest_model.pth"

# Save the model
torch.save(agent.q_network.state_dict(), model_path)
print(f"✅ Model saved successfully at {model_path}")

# Also log the model to Weights & Biases (wandb)
wandb.save(model_path)
