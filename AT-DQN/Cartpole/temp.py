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
DEBUG = True
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
if config['AT-DQN']['device']=='cpu':
    device = torch.device("cpu")

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

class StateAttentionTrackerLRU:
    def __init__(self, capacity, device):
        self.device=device
        self.capacity=capacity
        self.attention_values=(torch.ones(capacity, dtype=torch.float32, device=device))*config["AT-DQN"]["initial_attention"]
        self.current_index=0
        self.hash_to_index={} #dicitonary holding a state-attention pairs
        self.last_access = torch.zeros(capacity, dtype=torch.long, device=device) #pytorch tensor that stores the access time of each state (stored as index corresponding to hash_to_index)
        self.access_counter = 0  #global counter tracking when which state is accessed
        self.unique_state_counter=0
        self.old_state_counter=0
        self.old_action=0
        self.new_action=0

    def get_state_hash(self, state):
        # 1) Make sure we have a float32 tensor on GPU or CPU
        t = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        # 2) Cast to FP8 (E4M3)
        fp8 = t.to(torch.float8_e4m3fn)
        # 3) View the raw 8-bit bytes and hash
        raw_bytes = fp8.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return xxhash.xxh3_64(raw_bytes).hexdigest()
    

    # def get_state_hash(self, state):  # Function to hash state
    #     if isinstance(state, torch.Tensor):  
    #         if state.device != torch.device("cpu"):
    #             state_bytes = state.cpu().numpy().tobytes()
    #         else:
    #             state_bytes = state.numpy().tobytes()
    #     else:  # if array then
    #         state_bytes = np.asarray(state).tobytes()
    #     return xxhash.xxh3_64(state_bytes).hexdigest()
    
    def get_state_index(self, state, act=False):
        state_hash=self.get_state_hash(state)
        self.access_counter+=1

        if state_hash in self.hash_to_index: #if state already exists
            idx=self.hash_to_index[state_hash]
            self.last_access[idx]=self.access_counter
            if act==False:
                self.old_state_counter+=1
            return idx
    
        if act==False:
            if self.current_index<self.capacity: #if capacity not full and state doesn't exist
                idx=self.current_index
                self.hash_to_index[state_hash] = idx #map that index to the new state hassh
                self.last_access[idx]= self.access_counter #update last access of that index
                self.current_index+=1   #return the current_index
                self.unique_state_counter+=1
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
            self.unique_state_counter+=1

            return idx  # returns index after removing and adding new
        
        else:
            return torch.tensor(config["AT-DQN"]["initial_attention"])
    
    
    def batch_get_indices(self, states):  # vector index retrieval
        return torch.tensor(
            [self.get_state_index(state) for state in states],
            device=self.device,
            dtype=torch.long,
        )
    
    def batch_get_attention(self, states):  # vector get attention values for the retrieved indices
        indices = self.batch_get_indices(states)
        return self.attention_values[indices]
    
     #debugging code :
    def check_old_new(self, state):
        state_hash=self.get_state_hash(state)
        if state_hash in self.hash_to_index:
            return True
        else:
            return False
    
    def get_attention(self, state):  # get attention value for a single state index (used to act)
        exists=self.check_old_new(state)
        idx = self.get_state_index(state, act=True)
        if exists==True:
            self.old_action+=1
            #print("old state", self.attention_values[idx].item())
            return self.attention_values[idx]
        else:
            self.new_action+=1
            #print("new state", idx.item())
            return idx #this is the initial attention value directly returned by the get_state_index function
    
    
    def update_attention(self, states, weights):
        # Convert to tensor if needed - validate
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
        indices = self.batch_get_indices(states) #retrieve indices of states to be updated (in turn calls get_state_index to handle LRU)
        #print("old values",self.attention_values[indices])
        self.attention_values[indices]=weights
        #print("new values",self.attention_values[indices])
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
    
    # def compute_attention(self, td_errors):
    #    weights = torch.where(td_errors > config["AT-DQN"]["tau"], 0.9, 0.1)
    #    return weights
         
    def compute_attention(self, td_errors):
        weights=td_errors.abs() + 1e-6
        return weights
    
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
        self.attention_tracker=StateAttentionTrackerLRU(config["AT-DQN"]["LRU"], device)
        self.tau=config["AT-DQN"]["tau"]
        self.tau_start = 0.5        # threshold min
        self.tau_end   = 2.0        # threshold max
        
        
    def act(self, state, step_count):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        
        attention=self.attention_tracker.get_attention(state).item()

        frac = min(1.0, step_count / 150_000)

        threshold = self.tau_start + frac * (self.tau_end - self.tau_start)

        if attention > threshold: 
            self.exploration_count+=1 
            return np.random.randint(self.action_space), attention, threshold
        else:
            self.exploitation_count+=1
            return action_values.argmax(dim=1).item(), attention, threshold
            
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

        # for error in td_errors:
        #     if error.item()>30:
        #         print("LARGE!", error.item())
        
        attention_values=self.attention_tracker.compute_attention(td_errors)
        self.attention_tracker.update_attention(states, attention_values.squeeze())

        loss_fn = torch.nn.HuberLoss()
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
    
    def return_state_count(self):
    	return self.attention_tracker.unique_state_counter, self.attention_tracker.old_state_counter

    def return_action_count(self):
        return self.attention_tracker.new_action, self.attention_tracker.old_action

        
def train_agent(env_name, render=False):
    total_steps=config['AT-DQN']['T']
    lr=config['AT-DQN']['lr']
    
    total_reward = 0
    losses = []
    episode = 0
    episode_length = 0
    td_errors_per_episode = []
    q_values = []
    att_values=[]

    
    env = gym.make(env_name)
    state, _ = env.reset()
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = Agent(state_shape, action_size, lr)

    
    
    
    for step in tqdm(range(total_steps), desc="Training Progress"):
        episode_length += 1
        action, att, threshold = agent.act(state, step)
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.add_experience(state, action, reward, next_frame, done)
        result = agent.train_step()
        state = next_frame
        total_reward += reward
        if result is not None:
            loss, td_error, qvalue= result
            losses.append(loss)
            td_errors_per_episode.extend(td_error)
            q_values.extend(qvalue)
            att_values.append(att)

        if done:
            episode += 1
            mean_losses = np.mean(losses) if losses else 0.0
            mean_td_error = (
                np.mean(td_errors_per_episode) if td_errors_per_episode else 0.0
            )
            mean_q_value = np.mean(q_values) if q_values else 0.0
            mean_att_value = np.mean(att_values) if att_values else 0.0
            max_att_value = np.max(att_values) if att_values else 0.0
            min_att_value = np.min(att_values) if att_values else 0.0
            std_att_value = np.std(att_values) if att_values else 0.0
            unique_states, old_states=agent.return_state_count()
            new_actions, old_actions=agent.return_action_count()

            if DEBUG:
                wandb.log(
                    {
                        "global_step": step + 1,
                        "reward": total_reward,
                        "loss": mean_losses,
                        "episode_length": episode_length,
                        "mean_td_error": mean_td_error,
                        "mean_q_value": mean_q_value,
                        "mean attention" : mean_att_value,
                        "max attention" : max_att_value,
                        "min attention" : min_att_value,
                        "std attention" : std_att_value,
                        "No. of States Explored" : agent.exploration_count,
                        "No. of States Exploited" : agent.exploitation_count,
                        "Unique states added:" : unique_states,
                        "old states updated:" : old_states,
                        "Unique states (action):" : new_actions,
                        "old states (action):" : old_actions,
                        "theshold:" : threshold,
                    },
                    step=episode,
                )

            state, _ = env.reset()
            total_reward = 0
            losses = []
            episode_length = 0
            td_errors_per_episode = []
            q_values = []
            att_values=[]



    print("Training complete!")
    env.close()

    os.makedirs("AT_DQN_Models", exist_ok=True)
    torch.save(
        agent.q_network.state_dict(), f"AT_DQN_Models/ATDQN_{env_name.split('/')[-1]}_model.pth"
    )
    print(f"Model saved successfully!")

    if DEBUG:
        wandb.finish()
    else:
        print("Debug mode disabled, skipping wandb model upload")


if __name__ == "__main__":
    if DEBUG:
        wandb.init(
            project="AT-DQN",
            name="cartpole_ATDQN_anneal_lr",
            config={
                "total_steps": config['AT-DQN']['T'],
                "LRU": config['AT-DQN']['LRU'],
                "lr": config['AT-DQN']["lr"],
                "tau": config['AT-DQN']["tau"],
                "initial_attention": config['AT-DQN']["initial_attention"],
            },
        )
    else:
        print("Running in non-debug mode, wandb logging disabled")
        print(
            f"Config: {config['AT-DQN']['T']}, lr={config['AT-DQN']['lr']}"
        )

    # Train agent
    train_agent("CartPole-v1")
        

 
        
        
	    
           	
        
            
