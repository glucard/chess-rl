import torch, chess
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

def phi(observation, device):
    x = torch.tensor(observation, dtype=torch.float32, device=device)
    return x

def valid_move(board: chess.Board, move: chess.Move) -> bool:
    # check legal
    if not move in board.legal_moves:
        return False
    
    # check repetition
    """board.push(move)
    if board.is_repetition(count=4):
        board.pop()
        return False
    board.pop()"""

    # check move n repetition on last m moves
    if board.move_stack[-5:].count(move) > 1:
        return False

    # valid move
    return True

def valid_actions(board: chess.Board, all_moves: list):
    legal_moves = [i for i, move in enumerate(all_moves) if valid_move(board, move)]
    return legal_moves

def mask_invalid_actions(q_values, valid_actions):
    mask = torch.full_like(q_values, float('-inf'))
    mask[:, valid_actions] = 0
    y = q_values + mask
    return y

class DQN(nn.Module):
    def __init__(self, input_size, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 128)

        self.a = nn.Linear(128, action_space)
        self.v = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc6(x)

        a = self.a(x)
        v = self.v(x)
        q = v + a - a.mean()
        return q
    
    def q_train(self, target_net, optimizer, loss_fn, sample, gamma, replay_memory, device):
        optimizer.zero_grad()
        self.train()
        # states, actions, rewards, next_states, priorities, indices, weights = *zip(*sample), # let the ',' to not give syntax error
        states, actions, rewards, next_states, next_state_valid_actions, priorities, indices, weights = sample # let the ',' to not give syntax error
        
        BATCH_SIZE = len(states) # - 1

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        # weights = torch.stack(weights)
        
        
        non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        
        states, actions, rewards, non_final_next_states, weights = states.to(device), actions.to(device), rewards.to(device), non_final_next_states.to(device), weights.to(device)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_states_qvalues = target_net(non_final_next_states)
            q_values_masked = torch.cat([mask_invalid_actions(q.unsqueeze(0), va_i) for q, va_i in zip(next_states_qvalues, next_state_valid_actions)])
            next_state_values[non_final_states_mask] = q_values_masked.max(1)[0]
            
        # Set yj for terminal and non-terminal phij+1
        expected_next_action_values = rewards + gamma * next_state_values
        qvalues = self(states)
        qvalues = qvalues.gather(1, actions)

        # loss = loss_fn(qvalues, expected_next_action_values.unsqueeze(1))
        td_errors = torch.abs(qvalues - expected_next_action_values.unsqueeze(1).detach())
        # td_errors = torch.clamp(td_errors, min=-1, max=1)
        loss = torch.square(td_errors)  * weights.unsqueeze(1)

        loss = loss.mean()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        optimizer.step()
        
        replay_memory.update_priorities(indices, td_errors + 1e-5)
        return loss