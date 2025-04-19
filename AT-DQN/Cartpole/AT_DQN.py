import numpy as np
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import yaml
from datetime import timedelta
import pdb
import xxhash
DEBUG = False
if DEBUG:
    import wandb
    from wandb import AlertLevel

# Initilizations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['AT-DQN']['device']=='mps':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if config['AT-DQN']['device']=='cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

#defining the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(*state_shape, 64)  # cartpole input : (4,)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    

class ReplayBuffer:
    def __init__(
        self, capacity, device=device
    ):  
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


    #optimization - pinned memory for faster transfers
        # self.states = self.states.pin_memory()
        # self.actions = self.actions.pin_memory()
        # self.rewards = self.rewards.pin_memory()
        # self.next_states = self.next_states.pin_memory()
        # self.dones = self.dones.pin_memory()

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
    def update_attention_weights(self, indices, attention_values):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()  # Ensure indexing works on CPU tensors

        if attention_values.dim() == 1:
            attention_values = attention_values.unsqueeze(1)  # (batch_size, 1)

            self.attention_weight[indices] = attention_values.detach().cpu()

class StateAttentionTrackerLRU:
    def __init__(self, capacity, device, track_indices=None):
        self.device=device
        self.capacity=capacity
        self.attention_values=(torch.ones(capacity, dtype=torch.float32, device=device))*config["AT-DQN"]["initial_attention"]
        self.current_index=0
        self.hash_to_index={} #dicitonary holding a state-attention pairs
        self.last_access = torch.zeros(capacity, dtype=torch.long, device=device) #pytorch tensor that stores the access time of each state (stored as index corresponding to hash_to_index)
        self.access_counter = 0  #global counter tracking when which state is accessed
        self.track_indices = track_indices if track_indices is not None else []  # List of fixed indices to track attention

    def get_state_hash(self, state):  # Function to hash state
        if isinstance(state, torch.Tensor):  
            if state.device != torch.device("cpu"):
                state_bytes = state.cpu().numpy().tobytes()
            else:
                state_bytes = state.numpy().tobytes()
        else:  # if array then
            state_bytes = np.asarray(state).tobytes()
        return xxhash.xxh3_64(state_bytes).hexdigest()
    
    def get_state_index(self, state):
        state_hash=self.get_state_hash(state)
        self.access_counter+=1

        if state_hash in self.hash_to_index: #if state already exists
            idx=self.hash_to_index[state_hash]
            self.last_access[idx]=self.access_counter
            return idx
    
        if self.current_index<self.capacity: #if capacity not full and state doesn't exist
            idx=self.current_index
            self.current_index+=1   #return the current_index
            return idx
        
        else:  #if LRU full
            used_indices = torch.arange(self.current_index, device=self.device) #indices already used in LRU aranged from 0 to current_index
            idx = used_indices[
                torch.argmin(self.last_access[: self.current_index]) #Finds the index with minimum last access value
            ].item()
            old_hash = None
            for h, i in list(self.hash_to_index.items()):
                if i == idx:                #find the hash of state corresponding to least used index
                    old_hash = h
                    break
            if old_hash:
                del self.hash_to_index[old_hash] #delete the old hashed state

        self.hash_to_index[state_hash] = idx #new state's hash is assigned to that index
        self.last_access[idx] = self.access_counter #last access to current access counter

        return idx  # returns index after removing and adding new
    
    
    def batch_get_indices(self, states):  # vector index retrieval
        return torch.tensor(
            [self.get_state_index(state) for state in states],
            device=self.device,
            dtype=torch.long,
        )
    
    def batch_get_attention(self, states):  # vector get attention values for the retrieved indices
        indices = self.batch_get_indices(states)
        return self.attention_values[indices]
    
    def get_attention(self, state):  # get attention value for a single state index (used to act)
        idx = self.get_state_index(state)
        return self.attention_values[idx]
    
    def update_attention(self, states, weights):
        # Convert to tensor if needed - validate
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
        indices = self.batch_get_indices(states) #retrieve indices of states to be updated (in turn calls get_state_index to handle LRU)

        self.attention_values[indices]=weights
        #self.normalize_attention()
    
    def normalize_attention(self):
        if self.current_index == 0: #if empty, do nothing
            return
        used_values = self.attention_values[: self.current_index] #select only available values
        min_val = used_values.min() #compute min value amongst available
        max_val = used_values.max() #compute max value amongst available

        if min_val == max_val: #if min value and max value are same, do nothing
            return
        used_values.sub_(min_val).div_(max_val - min_val) #other min mas normalize
    
    def compute_attention(self, td_errors):
        weights = torch.where(td_errors > 0.4, 0.9, 0.1)
        return weights
         
    
    # def compute_attention(self, td_errors):
    #     weights=td_errors.abs() + 1e-6
    #     return weights
    
    def to(self, device):
        self.device = device
        self.attention_values = self.attention_values.to(device)
        self.last_access = self.last_access.to(device)
        return self


        

    
class Agent:
    def __init__(self, state_space, action_space, lr):
        self.state_space=state_space
        self.action_space=action_space
        self.q_network = QNetwork(state_space, action_space).to(device)
        self.target_network = QNetwork(state_space, action_space).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.lr=lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer=ReplayBuffer(config['AT-DQN']['replay_buffer'], device=device)
        self.gamma=config['AT-DQN']['gamma']
        self.exploration_count=0
        self.exploitation_count=0
        self.check_replay_size=config['AT-DQN']['warmup'] #warmup steps
        self.step_count=0
        fixed_states_indices = [0, 1, 2]
        self.attention_tracker=StateAttentionTrackerLRU(config["AT-DQN"]["LRU"], device, track_indices=fixed_states_indices)
        self.tau=config["AT-DQN"]["tau"]
        
    
        
    def act(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()

        if self.attention_tracker.get_attention(state).item() > self.tau: #CHECK! not using importance weight atm
            self.exploration_count+=1 
            return np.random.randint(self.action_space)
        else:
            self.exploitation_count+=1
            return action_values.argmax(dim=1).item()



    def train_step(self):
        if self.replay_buffer.size < self.check_replay_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(128)
        
        # optimization - gpu
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            max_next_q = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states).gather(1, actions.long())
        td_errors = targets - q_values
        
        attention_values=self.attention_tracker.compute_attention(td_errors)
        self.attention_tracker.update_attention(states, attention_values.squeeze())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        
        # add grad clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.step_count+=1
        if self.step_count%config['AT-DQN']['target_update']==0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return (
            loss.item(),
            td_errors.abs().squeeze().cpu().tolist(),
            q_values.squeeze().cpu().tolist(),
        )
        
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)        
        
def train_agent(env_name, render=False):
    total_steps=config['AT-DQN']['T']
    lr=config['AT-DQN']['lr']
    
    total_reward = 0
    losses = []
    episode = 0
    episode_length = 0
    td_errors_per_episode = []
    q_values = []
    
    env = gym.make(env_name)
    state, _ = env.reset()
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = Agent(state_shape, action_size, lr)

    
    
    
    for step in tqdm(range(total_steps), desc="Training Progress"):
        episode_length += 1
        action = agent.act(state)
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
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
                        "episode_length": episode_length,
                        "mean_td_error": mean_td_error,
                        "mean_q_value": mean_q_value,
                        "No. of States Explored" : agent.exploration_count,
                        "No. of States Exploit" : agent.exploitation_count,
                    },
                    step=episode,
                )


            # if episode % 100 == 0:
            #     if DEBUG:
            #         wandb.log(
            #             {

            #             },
            #             step=episode,
            #         )
            #     else:
            #         print(
            #             f"Episode {episode} - Explored: {agent.exploration_count}, Exploited: {agent.exploitation_count}"
            #         )
            #         print(
            #         f"Episode {episode} - Steps: {step+1}, Reward: {total_reward:.2f}, Loss: {mean_losses:.4f}, "
            #         f"Q Value: {mean_q_value:.4f}"
            #     )


            # agent.exploration_count=0
            # agent.exploitation_count=0
            state, _ = env.reset()
            total_reward = 0
            losses = []
            episode_length = 0
            td_errors_per_episode = []
            q_values = []


    print("Training complete!")
    env.close()

    os.makedirs("AT_DQN_Models", exist_ok=True)
    torch.save(
        agent.q_network.state_dict(), f"AT_DQN_Models/ATDQN_{env_name.split('/')[-1]}_model.pth"
    )
    print(f"Model saved successfully!")

    if DEBUG:

        
                #wandb.save(model_path)
        wandb.finish()
    else:
        print("Debug mode disabled, skipping wandb model upload")


if __name__ == "__main__":
    if DEBUG:
        wandb.init(
            project="AT-DQN",
            name="cartpole_ATDQN_v2",
            config={
                "total_steps": config['AT-DQN']['T'],
                "LRU": config['AT-DQN']['LRU'],
                "lr": config['AT-DQN']["lr"],
            },
        )
    else:
        print("Running in non-debug mode, wandb logging disabled")
        print(
            f"Config: {config['AT-DQN']['T']}, lr={config['AT-DQN']['lr']}"
        )

    # Train agent
    train_agent("CartPole-v1")
        

 
        
        
	    
           	
        
            
