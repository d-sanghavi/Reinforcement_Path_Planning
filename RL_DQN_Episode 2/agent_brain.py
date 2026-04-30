import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

GRID_SIZE = 15
STATE_SIZE = GRID_SIZE * GRID_SIZE * 2  # 450 — agent pos (225) + goal pos (225)
ACTION_SIZE = 4

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, actions, learning_rate=0.001, reward_decay=0.98,
                 e_greedy=1.0, e_greedy_min=0.01, e_greedy_decay=0.995,
                 batch_size=64, memory_size=10000, target_update_freq=100):

        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_min = e_greedy_min
        self.epsilon_decay = e_greedy_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Online and target networks
        self.online_net = DQNNetwork()
        self.target_net = DQNNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        # state is a tuple: ((agent_row, agent_col), (goal_row, goal_col))
        agent_pos, goal_pos = state
        one_hot = np.zeros(STATE_SIZE, dtype=np.float32)
        # First 225 dims = agent position
        agent_idx = agent_pos[0] * GRID_SIZE + agent_pos[1]
        one_hot[agent_idx] = 1.0
        # Next 225 dims = goal position
        goal_idx = GRID_SIZE * GRID_SIZE + goal_pos[0] * GRID_SIZE + goal_pos[1]
        one_hot[goal_idx] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)  # shape (1, 450)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        with torch.no_grad():
            q_values = self.online_net(self.state_to_tensor(state))
        return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Build tensors
        state_tensors = torch.cat([self.state_to_tensor(s) for s in states])         # (64, 225)
        next_state_tensors = torch.cat([self.state_to_tensor(s) for s in next_states])  # (64, 225)
        action_tensor = torch.LongTensor(actions)
        reward_tensor = torch.FloatTensor(rewards)
        done_tensor = torch.FloatTensor(dones)

        # Current Q values from online net
        q_values = self.online_net(state_tensors)                          # (64, 4)
        q_predict = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  # (64,)

        # Target Q values from target net
        with torch.no_grad():
            q_next = self.target_net(next_state_tensors).max(1)[0]         # (64,)
        q_target = reward_tensor + self.gamma * q_next * (1 - done_tensor)

        # Loss and backprop
        loss = self.loss_fn(q_predict, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network hard update
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def plot_results(self, steps, rewards, losses):
        import os
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via Steps')

        ax2.plot(np.arange(len(rewards)), rewards, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Episode via Reward')

        ax3.plot(np.arange(len(losses)), losses, 'g')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Episode via Loss')

        plt.tight_layout()
        
        # Save to Results folder
        os.makedirs('Results', exist_ok=True)
        plt.savefig('Results/training_results.png')
        
        # Show plot (this blocks until the graph window is closed)
        plt.show()