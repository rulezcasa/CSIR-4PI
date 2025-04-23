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

class AttentionNet(nn.Module): #attention network
    def __init__(self, state_shape):
        super(AttentionNet, self).__init__()
        self.fc1=nn.Linear(*state_shape, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        attention=self.sigmoid(self.fc3(x))
        return attention
    
    
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
        self.tau=config["AT-DQN"]["tau"]
        self.Attnet=AttentionNet(state_space).to(device)
        self.att_optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
    
        
    def act(self, state):
        if self.replay_buffer.size < self.check_replay_size:
            attention=0.9
        
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state_tensor)
            self.q_network.train()

            self.Attnet.eval()
            with torch.no_grad():
                attention=self.Attnet(state_tensor)
            self.Attnet.train()

        if attention > self.tau: 
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
        
        td_errors_detached=td_errors.detach().abs()
        scale=10
        normalized = torch.sigmoid(td_errors_detached / scale)  # range (0,1)
        binary_labels = (normalized > 0.4).float()   
        attention_values=self.Attnet(states)
        att_loss_fn=torch.nn.BCELoss()
        att_loss=att_loss_fn(attention_values, binary_labels)
        self.att_optimizer.zero_grad()
        att_loss.backward()
        self.att_optimizer.step()

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
            att_loss.item(),
            attention_values.detach().cpu().tolist(),
            normalized.squeeze().cpu().tolist()
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
    att_losses=[]
    att_values=[]
    normalized_tds=[]
    
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
            loss, td_error, qvalue, att_loss, att_value, normalized_td = result
            losses.append(loss)
            td_errors_per_episode.extend(td_error)
            q_values.extend(qvalue)
            att_losses.append(att_loss)
            att_values.extend(att_value)
            normalized_tds.extend(normalized_td)

        if done:
            episode += 1
            mean_losses = np.mean(losses) if losses else 0.0
            mean_td_error = (
                np.mean(td_errors_per_episode) if td_errors_per_episode else 0.0
            )
            mean_q_value = np.mean(q_values) if q_values else 0.0
            mean_att_losses = np.mean(att_losses) if att_losses else 0.0
            mean_att_values=np.mean(att_values) if att_values else 0.0
            mean_norm_td=np.mean(normalized_tds) if att_values else 0.0



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
                        "Attention network loss": mean_att_losses,
                        "Mean Attention values": mean_att_values,
                        "Mean Norm TD valules": mean_norm_td,
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
            name="cartpole_ATDQN_AttNet",
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
        

 
        
        
	    
           	
        
            
