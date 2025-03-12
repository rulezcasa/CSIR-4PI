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

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, "seaquest_training_log.txt")
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)   

# Setup system usage log
sys_logger = logging.getLogger("system")
sys_logger.addHandler(logging.FileHandler("logs/seaquest_usage_log.txt", mode="w"))
sys_logger.setLevel(logging.INFO)

# Redirect stdout and stderr to log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)  # Capture errors too

# Function to log CPU & GPU usage
def log_system_usage():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu = torch.cuda.utilization(0)
        gpu_mem = torch.cuda.memory_allocated(0) / 1e6  # MB
        sys_logger.info(f"CPU: {cpu}%, RAM: {ram}%, GPU: {gpu}%, GPU Mem: {gpu_mem:.2f} MB")
    else:
        sys_logger.info(f"CPU: {cpu}%, RAM: {ram}% (No GPU)")

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
        
        
        # Attention-based DQN Agent
class ATDQNAgent:
    def __init__(self, action_size, state_shape, tau=0.2, beta_start=0.4, beta_end=1.0, T=1000, device="cuda"):
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_shape, action_size).to(device)
        self.target_network = QNetwork(state_shape, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99  

        # Attention Mechanism
        self.alpha = defaultdict(lambda: 1.0)  
        self.td_errors = defaultdict(list)

        # Exploration threshold
        self.tau = tau  

        # Beta Annealing
        self.beta = beta_start
        self.beta_end = beta_end
        self.delta_beta = (beta_end - beta_start) / T  


    def get_state_hash(self, state):
        state_bytes = state.tobytes()  # Convert state array to bytes
        return hashlib.sha256(state_bytes).hexdigest()
        
    def get_attention(self, state):
        state_hash = self.get_state_hash(state)
        return self.alpha[state_hash]

    def update_attention(self, state, td_error):
        state_hash = self.get_state_hash(state)
        self.td_errors[state_hash].append(abs(td_error))
        self.alpha[state_hash] = np.mean(self.td_errors[state_hash])

    def normalize_attention(self):
        max_alpha = max(self.alpha.values(), default=1)
        for state_hash in self.alpha:
            self.alpha[state_hash] /= max_alpha  

    def compute_importance_weight(self, state, N):
        state_hash = self.get_state_hash(state)
        alpha_s = self.alpha[state_hash]
        return (1 / (N * alpha_s)) ** self.beta

    def act(self, state):
        sigma = self.get_attention(state)
        if sigma > self.tau:
            return np.random.choice(self.action_size)  
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, batch_size=32):
            
        if self.replay_buffer.size() < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(int)).to(self.device)

        target_q_values = self.target_network(next_states).detach()
        max_next_q = target_q_values.max(dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states)
        q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze()

        td_errors = targets - q_values

        for i in range(batch_size):
            self.update_attention(states[i].cpu().numpy(), td_errors[i].item())

        # Importance sampling correction
        N = len(self.replay_buffer.buffer)
        importance_weights = torch.FloatTensor(
            [self.compute_importance_weight(states[i].cpu().numpy(), N) for i in range(batch_size)]
        ).to(self.device)

        importance_weights /= importance_weights.max()  

        loss = torch.mean(importance_weights * (td_errors ** 2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def anneal_beta(self):
        self.beta = min(self.beta + self.delta_beta, self.beta_end)
        
        


# Initialize Weights & Biases
wandb.init(project="AT-DQN", name="Seaquest",config={"episodes": 1000, "batch_size": 32})
logging.info("Training process initialized.")

# Step 1: Check if environment is created
try:
    env = gym.make('ALE/Seaquest-v5')
    print("Environment created successfully!")
except Exception as e:
    print(f"Error in creating environment: {e}")
    exit()

obs = env.reset()
state_shape = (4, 84, 84)  # 4 stacked frames, 84x84 resolution
action_size = env.action_space.n

# Initialize agent
agent = ATDQNAgent(action_size, state_shape, device="cuda" if torch.cuda.is_available() else "cpu")

num_episodes = 1000
batch_size = 32
reward_memory = deque(maxlen=100)  # Stores the last 100 rewards
total_cumulative_reward = 0  # Initialize cumulative reward
td_error=[]

reward_td_table = wandb.Table(columns=["Total Reward", "Mean TD Error"])

for episode in tqdm(range(num_episodes), desc="Episodic Progress", unit="episode"):
    print(f"Episode {episode + 1}/{num_episodes} started...")

    state, info = env.reset()
    state = preprocess_frame(state)
    state_stack = np.stack([state] * 4, axis=0)  # Expected: (4, 84, 84)
    total_reward = 0
    loss = 0.0
    beta = agent.beta  # Assuming ATDQNAgent has a beta parameter
    td_error=[]

    for step in range(10000):
        action = agent.act(state_stack)
        try:
            next_frame, reward, done, _, _ = env.step(action)
        except Exception as e:
            print(f"Error in env.step(): {e}")
            break

        next_frame = preprocess_frame(next_frame)
        next_state_stack = np.concatenate((state_stack[1:], np.expand_dims(next_frame, axis=0)), axis=0)

        agent.replay_buffer.add((state_stack, action, reward, next_state_stack, done))
        loss = agent.train(batch_size)

        state_stack = next_state_stack
        total_reward += reward
        td_error.append(loss)



        if done:
            print("Episode completed!")
            break
        
    total_cumulative_reward += total_reward
    reward_memory.append(total_reward)  # Store reward for averaging
    
    mean_td_error = np.mean([x for x in td_error if x is not None]) if td_error else 0.0
    
    reward_td_table.add_data(total_reward, mean_td_error)
    
    logging.info(f"Episode {episode + 1}: Reward = {total_reward}, Loss = {loss:.4f}, Beta = {beta:.4f}")

    # Log episodic metrics to wandb
    wandb.log({
        "Beta per episode": beta,
        "Mean TD error per episode": mean_td_error,
        "Reward per episode": total_reward
    })


    print(f"Episode {episode + 1}: Reward = {total_reward}, Loss = {loss:.4f}, Beta = {beta:.4f}")

    agent.anneal_beta()

    # Update target network every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.update_target_network()
        print(f"✅ Updated target network at episode {episode + 1}")
        log_system_usage()

    # Logs every 100 episodes
    if (episode + 1) % 100 == 0:
        wandb.log({
            "Beta per 100 episodes": beta,
            "Reward per 100 episodes": total_reward,
            "Average reward over last 100 episodes": np.mean(reward_memory),
            "Total Reward vs Mean TD Error- 100 ep": wandb.plot.scatter(reward_td_table, "Total Reward", "Mean TD Error", title="Reward vs TD Error - 100 ep")
        })
        
wandb.log({"Total Reward vs Mean TD Error- final": wandb.plot.scatter(reward_td_table, "Total Reward", "Mean TD Error", title="Reward vs TD Error - final")})
       
      
logging.info("Training completed successfully.")
env.close()
wandb.finish()



# Define the path to save the model
model_path = "AT_DQN_Models/atdqn_seaquest_model.pth"

# Save the model
torch.save(agent.q_network.state_dict(), model_path)
print(f"✅ Model saved successfully at {model_path}")

# Also log the model to Weights & Biases (wandb)
wandb.save(model_path)

sys.stdout.log.close()
sys.stderr.log.close()
