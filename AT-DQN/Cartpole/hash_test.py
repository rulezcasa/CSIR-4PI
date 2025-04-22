import xxhash
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device='cpu'

def get_state_hash(state):  # Function to hash state
    if isinstance(state, torch.Tensor):  
        if state.device != torch.device("cpu"):
            state_bytes = state.cpu().numpy().tobytes()
        else:
            state_bytes = state.numpy().tobytes()
    else:  # if array then	
        state_bytes = np.asarray(state).tobytes()
    return xxhash.xxh3_64(state_bytes).hexdigest()

env = gym.make("CartPole-v1")
state, _ = env.reset()
print("raw state is:", repr(state))
state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
print("state tensor is:", state_tensor)
state_bytes = state_tensor.cpu().numpy().tobytes()
print("state bytes is:", state_bytes)
print("hash computed is:", get_state_hash(state_tensor))

print("")

np.save("state.npy", state)
manual_state = np.load("state.npy")
print("manual state is:", manual_state)
manual_state_tensor=torch.from_numpy(manual_state).float().unsqueeze(0).to(device)
print("manual state tensor is:", manual_state_tensor)
manual_state_bytes = manual_state_tensor.cpu().numpy().tobytes()
print("manual state bytes is:", manual_state_bytes)
print("manual hash computed is:", get_state_hash(manual_state_tensor))

print("")


print("Original state dtype:", state.dtype)
print("State tensor dtype:", state_tensor.dtype)
print("Manual state dtype:", manual_state.dtype)
print("Manual state tensor dtype:", manual_state_tensor.dtype)


    

