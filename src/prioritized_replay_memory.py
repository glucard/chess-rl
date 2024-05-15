import torch
import numpy as np
from collections import deque

class PrioritizedReplayMemory:
    
    def __init__(self, capacity, torch_device, prob_alpha=0.6):

        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.prob_alpha = prob_alpha
        self.torch_device = torch_device

    def __len__(self):
        return len(self.memory)
    
    def max_priority(self):
        states, actions, rewards, next_states, next_state_valid_actions, priorities = *zip(*self.memory), # let the ',' to not give syntax error
        return max(priorities)

    def push(self, state, action, reward, next_state, next_state_valid_actions):
        max_prio  = self.max_priority() if len(self) > 0 else 1.0
        experience = (state, action, reward, next_state, next_state_valid_actions, max_prio)
        self.memory.append(experience)
    
    def update_priorities(self, indices, new_priorities):
        states, actions, rewards, next_states, next_state_valid_actions, priorities = *zip(*self.memory), # let the ',' to not give syntax error

        for idx, prio in zip(indices, new_priorities):
            prio = prio.item()
            self.memory[idx]= states[idx], actions[idx], rewards[idx], next_states[idx], next_state_valid_actions[idx], prio

    def sample(self, batch_size, beta=0.4): # maybe separate by episode to avoid sequence where final_state -> start_state
        states, actions, rewards, next_states, next_state_valid_actions, priorities = *zip(*self.memory), # let the ',' to not give syntax error
        priorities = np.array(priorities)
        probs = priorities ** self.prob_alpha
        probs /= sum(probs)

        m = len(self)

        indices = np.random.choice(m, batch_size, p=probs)

        samples = [self.memory[idx] for idx in indices]
        
        weights = (m * probs[indices] ** (-beta))
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.torch_device)

        states, actions, rewards, next_states, next_state_valid_actions, priorities = *zip(*samples),
        return states, actions, rewards, next_states, next_state_valid_actions, priorities, indices, weights
    
    def clear(self):
        self.memory.clear()