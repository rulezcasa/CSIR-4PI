import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from datetime import timedelta
import pdb

device="mps"

# Defining the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, 64)  # cartpole input : (4,)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


env = gym.make("CartPole-v1")  # Use "human" to render
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = QNetwork(state_space, action_space).to(device)
model.load_state_dict(torch.load(
    "/Volumes/Harish/CSIR Research/Codespace/CSIR-4PI/AT-DQN/Cartpole/AT_DQN_Models/ATDQN_CartPole-v1_model.pth",
    map_location=torch.device(device)
))
model.eval()

total_rewards = []

# Running for 100 episodes with tqdm for progress tracking
for episode in tqdm(range(100), desc="Running episodes"):
    state = env.reset()[0]  # for gymnasium; use `state = env.reset()` for older gym
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    total_rewards.append(total_reward)

# Print the average reward over 100 episodes
average_reward = np.mean(total_rewards)
print(f"Average reward over 100 episodes: {average_reward}")
