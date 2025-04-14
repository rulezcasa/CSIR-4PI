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
DEBUG = True
if DEBUG:
    import wandb
    from wandb import AlertLevel

# Initilizations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['Vanilla-DQN']['device']=='mps':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if config['Vanilla-DQN']['device']=='cuda':
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

    # #optimization - pinned memory for faster transfers
    #     self.states = self.states.pin_memory()
    #     self.actions = self.actions.pin_memory()
    #     self.rewards = self.rewards.pin_memory()
    #     self.next_states = self.next_states.pin_memory()
    #     self.dones = self.dones.pin_memory()

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
    
class Agent:
    def __init__(self, state_space, action_space, lr):
        self.state_space=state_space
        self.action_space=action_space
        self.q_network = QNetwork(state_space, action_space).to(device)
        self.target_network = QNetwork(state_space, action_space).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.lr=lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer=ReplayBuffer(config['Vanilla-DQN']['replay_buffer'], device=device)
        self.gamma=config['Vanilla-DQN']['gamma']
        self.exploration_count=0
        self.exploitation_count=0
        self.check_replay_size=config['Vanilla-DQN']['warmup'] #warmup steps
        self.step_count=0
        
    def act(self, state, epsilon):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()

        if np.random.random() > epsilon: #exploit
            self.exploitation_count+=1        	
            return action_values.argmax(dim=1).item()
        else:
            self.exploration_count+=1 
            return np.random.randint(self.action_space)
            
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
        loss = torch.mean(td_errors**2)
        self.optimizer.zero_grad()
        loss.backward()
        
        # add grad clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.step_count+=1
        if self.step_count%config['Vanilla-DQN']['target_update']==0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return (
            loss.item(),
            td_errors.abs().squeeze().cpu().tolist(),
            q_values.squeeze().cpu().tolist(),
        )
        
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)        
        
def train_agent(env_name, render=False):
    total_steps=config['Vanilla-DQN']['T']
    epsilon_start=config['Vanilla-DQN']['epsilon_start']
    epsilon_end=config['Vanilla-DQN']['epsilon_end']
    decay_steps=config['Vanilla-DQN']['decay_steps']
    lr=config['Vanilla-DQN']['lr']
    epsilon=epsilon_start
    
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
        if step < decay_steps:
            epsilon = epsilon_start - (step / decay_steps) * (epsilon_start - epsilon_end)       
        else:
            epsilon = epsilon_end
        action = agent.act(state,epsilon)
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
                        "epsilon":epsilon,
                    },
                    step=episode,
                )
            #else:
            #    print(
            #        f"Episode {episode} - Steps: {step+1}, Reward: {total_reward:.2f}, Loss: {mean_losses:.4f}, "
            #        f"Q Value: {mean_q_value:.4f}"
            #    )


            if episode % 100 == 0:
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
                    print(
                    f"Episode {episode} - Steps: {step+1}, Reward: {total_reward:.2f}, Loss: {mean_losses:.4f}, "
                    f"Q Value: {mean_q_value:.4f}"
                )


                agent.exploration_count = 0
                agent.exploitation_count = 0

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
        agent.q_network.state_dict(), f"AT_DQN_Models/atdqn_{env_name.split('/')[-1]}_model.pth"
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
            name="cartpole_vanilla_v4",
            config={
                "total_steps": config['Vanilla-DQN']['T'],
                "epsilon_start": config['Vanilla-DQN']['epsilon_start'],
                "epsilon_end": config['Vanilla-DQN']['epsilon_end'],
                "lr": config['Vanilla-DQN']["lr"],
            },
        )
    else:
        print("Running in non-debug mode, wandb logging disabled")
        print(
            f"Config: {config['Vanilla-DQN']['T']}, {config['Vanilla-DQN']['epsilon_start']}, beta_end={config['Vanilla-DQN']['epsilon_end']}, lr={config['Vanilla-DQN']['lr']}"
        )

    # Train agent
    train_agent("CartPole-v1")
        
        
        
    

 
        
        
	    
           	
        
            
