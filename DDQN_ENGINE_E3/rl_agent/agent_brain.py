"""
agent_brain.py
──────────────
Double Deep Q-Network (DDQN) agent for grid path planning.

Architecture:
  - Online Network: FC(state_size→256→128→action_size)
  - Target Network: identical, weights synced every TARGET_UPDATE_FREQ steps
  - Experience Replay Buffer: deque with 50K capacity
  - Training: Adam, γ=0.99, ε-greedy (1.0 → 0.01 over training)
  - Inference: fully greedy (ε=0), deterministic

The "Double" in DDQN:
  - Action SELECTION uses online network
  - Action EVALUATION uses target network
  → Reduces Q-value overestimation vs vanilla DQN
"""

import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
GAMMA = 0.99            # discount factor
LR = 5e-4               # Adam learning rate
BATCH_SIZE = 64         # replay batch size
REPLAY_CAPACITY = 50_000
TARGET_UPDATE_FREQ = 500  # steps between target network sync
EPS_START = 1.0         # initial exploration rate
EPS_END = 0.01          # minimum exploration rate
EPS_DECAY_STEPS = 50_000  # steps for epsilon to decay from start to end
MIN_REPLAY_SIZE = 512   # minimum replay size before learning starts


# ═══════════════════════════════════════════════════════════════════════════════
# Q-NETWORK (shared by online and target)
# ═══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Dueling Q-Network Architecture.
    Splits the network into a Value stream and an Advantage stream.
    """

    def __init__(self, state_size: int, action_size: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(inplace=True)
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, action_size)
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """A binary tree data structure where the parent's value is the sum of its children."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, s: float):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        if data_idx >= self.n_entries:
            data_idx = max(0, self.n_entries - 1)
            leaf_idx = data_idx + self.capacity - 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """Experience Replay Buffer using a SumTree for Prioritized Experience Replay (PER)."""
    
    def __init__(self, capacity: int = REPLAY_CAPACITY, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01  # small amount to avoid zero priority

    def push(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
            
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        )
        self.tree.add(max_p, transition)

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        batch = []
        idxs = []
        segment = self.tree.total_p / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            
            p = max(self.epsilon, p)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_p
        sampling_probabilities = np.maximum(sampling_probabilities, 1e-8)
        
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
            idxs,
            torch.tensor(is_weight,              dtype=torch.float32).unsqueeze(1)
        )

    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DDQNAgent:
    """
    Double DQN agent.

    Parameters
    ----------
    state_size : int
        Dimensionality of the state vector.
    action_size : int
        Number of discrete actions.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        # ── Networks ──────────────────────────────────────────────────────────
        self.online_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        # ── Replay buffer ──────────────────────────────────────────────────────
        self.replay = PrioritizedReplayBuffer(REPLAY_CAPACITY)

        # ── Training state ─────────────────────────────────────────────────────
        self.global_step = 0
        self.epsilon = EPS_START

        # ── Metrics (for dashboard) ────────────────────────────────────────────
        self.episode_rewards: list = []
        self.episode_steps: list = []
        self.losses: list = []
        self.epsilons: list = []

    # ── Epsilon Schedule ──────────────────────────────────────────────────────

    def _update_epsilon(self):
        """Linear epsilon decay from EPS_START to EPS_END over EPS_DECAY_STEPS."""
        t = min(self.global_step, EPS_DECAY_STEPS)
        self.epsilon = EPS_START + (EPS_END - EPS_START) * (t / EPS_DECAY_STEPS)

    # ── Action Selection ──────────────────────────────────────────────────────

    def select_action(
        self, 
        state: np.ndarray, 
        greedy: bool = False,
        current_dist: float = None,
        max_dist: float = None
    ) -> int:
        """
        Select an action.

        Parameters
        ----------
        state : np.ndarray
            Current environment state.
        greedy : bool
            If True, always select best action (no exploration).
            Set True at inference time for deterministic output.
        current_dist : float, optional
            Current distance to goal (used to scale epsilon).
        max_dist : float, optional
            Max distance to goal (used to scale epsilon).
        """
        # Dynamic epsilon scaling: explore more when far, exploit more when close
        effective_epsilon = self.epsilon
        if not greedy and current_dist is not None and max_dist is not None and max_dist > 0:
            dist_ratio = min(1.0, current_dist / max_dist)
            # scale base epsilon by distance ratio (meaning closer = smaller epsilon)
            effective_epsilon = self.epsilon * dist_ratio
            
        if not greedy and random.random() < effective_epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    # ── Learning Update ───────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        """
        Sample a batch from prioritized replay and perform one DDQN update.
        """
        if len(self.replay) < MIN_REPLAY_SIZE:
            return None

        # Increase beta dynamically from 0.4 to 1.0 during training for IS weights
        beta = min(1.0, 0.4 + (self.global_step / 100000.0) * (1.0 - 0.4))
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(BATCH_SIZE, beta=beta)
        
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)
        weights     = weights.to(self.device)

        # ── Double DQN update ─────────────────────────────────────────────────
        # 1. Online net selects the BEST NEXT ACTION
        with torch.no_grad():
            online_next_q = self.online_net(next_states)
            best_actions = online_next_q.argmax(dim=1, keepdim=True)

            # 2. Target net EVALUATES that action
            target_next_q = self.target_net(next_states)
            max_next_q = target_next_q.gather(1, best_actions).squeeze(1)

            # 3. Bellman target
            target_q = rewards + GAMMA * max_next_q * (1.0 - dones)

        # 4. Online net Q-value for taken actions
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 5. TD Errors for Prioritized Replay Update
        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        # 6. Weighted Huber loss
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (weights.squeeze(1) * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        loss_val = weighted_loss.item()
        self.losses.append(loss_val)

        # ── Target network sync ───────────────────────────────────────────────
        if self.global_step % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug(f"[DDQN] Target network synced at step {self.global_step}")

        return loss_val

    # ── Experience Storage ────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer and update step counter."""
        self.replay.push(state, action, reward, next_state, done)
        self.global_step += 1
        self._update_epsilon()

    # ── Model Persistence ─────────────────────────────────────────────────────

    def save(self, path: str):
        """Save online network weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "global_step": self.global_step,
            "epsilon": self.epsilon,
        }, path)
        logger.info(f"[DDQN] Saved weights to {path}")

    def load(self, path: str):
        """Load online + target network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.epsilon = checkpoint.get("epsilon", EPS_END)
        self.online_net.eval()
        self.target_net.eval()
        logger.info(f"[DDQN] Loaded weights from {path} (step={self.global_step})")

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.online_net.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# PPO AGENT (Actor-Critic)
# ═══════════════════════════════════════════════════════════════════════════════

from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_size)
        )
    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)

class CriticNetwork(nn.Module):
    def __init__(self, state_size: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]

class PPOAgent:
    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = RolloutBuffer()
        
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.train_iters = 10
        self.global_step = 0
        
        self.actor_losses = []
        self.critic_losses = []
        # Expose epsilon for API compatibility with DDQN in run_agent.py (even if unused)
        self.epsilon = 0.0

    def select_action(self, state: np.ndarray, greedy: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(state_tensor)
            val = self.critic(state_tensor)
        
        if greedy:
            action = dist.probs.argmax(dim=-1).item()
            logprob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        else:
            action = dist.sample().item()
            logprob = dist.log_prob(torch.tensor(action).to(self.device)).item()
            
        return action, logprob, val.item()

    def remember(self, state, action, logprob, reward, value, done):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.rewards.append(reward)
        self.buffer.values.append(value)
        self.buffer.dones.append(done)
        self.global_step += 1

    def learn(self):
        if len(self.buffer.states) == 0:
            return None, None
            
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        old_values = torch.FloatTensor(self.buffer.values).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones).to(self.device)
        
        # Generalized Advantage Estimation (GAE)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_val = 0 if (t == len(rewards)-1 or dones[t]) else old_values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t]) - old_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * last_gae
            returns[t] = advantages[t] + old_values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        a_loss_sum, c_loss_sum = 0, 0
        
        # PPO Update epochs
        for _ in range(self.train_iters):
            dist = self.actor(states)
            values = self.critic(states)
            
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(values, returns)
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            
            a_loss_sum += actor_loss.item()
            c_loss_sum += critic_loss.item()
            
        self.buffer.clear()
        
        avg_a_loss = a_loss_sum / self.train_iters
        avg_c_loss = c_loss_sum / self.train_iters
        self.actor_losses.append(avg_a_loss)
        self.critic_losses.append(avg_c_loss)
        return avg_a_loss, avg_c_loss

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)