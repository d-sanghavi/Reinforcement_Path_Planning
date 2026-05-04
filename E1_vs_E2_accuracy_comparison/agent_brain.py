# agent_brain.py — Combined agent brain: Q-Learning (E1) + DQN (E2)
#
# QLearningTable  — tabular Bellman update, Episode-1 algorithm
# DQNAgent        — neural network with replay buffer, Episode-2 algorithm
#
# Both agents expose:
#   choose_action(state)
#   learn(...)
#   print_q_table()   ← prints to terminal in Episode-2 console style
#   plot_results(...) ← per-agent subplot figure
#
# plot_comparison() at the bottom generates the final side-by-side
# comparison figure with three subplots:
#   Episode vs Steps  |  Episode vs Reward  |  Episode vs Loss

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# Use the non-interactive Agg backend so matplotlib never opens a tkinter
# window that would conflict with the running pygame event loop.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import random
import os

from env import final_states   # reads the global best-route dict from env.py

# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE 1 — Tabular Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

class QLearningTable:
    """
    Tabular Q-Learning (Episode 1).

    Parameters
    ----------
    actions       : list[int]  — [0, 1, 2, 3]
    learning_rate : float      — alpha
    reward_decay  : float      — gamma
    e_greedy      : float      — epsilon (constant; no decay in E1)
    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr      = learning_rate
        self.gamma   = reward_decay
        self.epsilon = e_greedy

        self.q_table       = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # ── Action selection ──────────────────────────────────────────────────── #
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            return state_action.idxmax()
        return np.random.choice(self.actions)

    # ── Bellman update ────────────────────────────────────────────────────── #
    def learn(self, state, action, reward, next_state):
        """Returns the updated Q-value (accumulated as 'cost' in run_agent)."""
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if next_state != 'goal' and next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)
        return self.q_table.loc[state, action]

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.DataFrame(
                [[0] * len(self.actions)],
                columns=self.q_table.columns,
                index=[state])
            self.q_table = pd.concat([self.q_table, new_row])

    # ── Terminal display ──────────────────────────────────────────────────── #
    def print_q_table(self):
        """
        Print the Q-table in Episode-2 console style:
        • Final-route sub-table (states along the best path)
        • Full Q-table (all visited states)
        """
        e = final_states()          # {0: (r,c), 1: (r,c), ...}

        # Build final-route sub-table
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)
        for i in range(len(e)):
            state = str(e[i])       # e.g. "(3, 5)"
            if state in self.q_table.index:
                self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        sep = "=" * 72
        hdr = "-" * 72

        print()
        print(sep)
        print("  [Q-LEARNING — E1]  FINAL ROUTE Q-TABLE  (states on best path)")
        print(sep)
        print(f"  States on best path : {len(self.q_table_final.index)}")
        print(hdr)
        if len(self.q_table_final.index) > 0:
            # Pretty-print with action header
            col_labels = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
            renamed = self.q_table_final.rename(columns=col_labels)
            print(renamed.to_string())
        else:
            print("  (no states recorded — agent may not have reached the goal)")
        print(sep)

        print()
        print(sep)
        print("  [Q-LEARNING — E1]  FULL Q-TABLE  (all visited states)")
        print(sep)
        print(f"  Total states visited : {len(self.q_table.index)}")
        print(hdr)
        col_labels = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
        renamed = self.q_table.rename(columns=col_labels)
        print(renamed.to_string())
        print(sep)
        print()

    # ── Per-agent plots (3 subplots, Episode-2 style) ────────────────────── #
    def plot_results(self, steps, cost):
        os.makedirs("Results", exist_ok=True)

        window   = max(1, len(steps) // 20)
        smoothed = np.convolve(steps, np.ones(window) / window, mode='valid')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Episode 1 — Q-Learning Training Results",
                     fontsize=13, fontweight='bold')

        ax1.plot(np.arange(len(steps)),    steps,    'b', lw=0.8, alpha=0.8)
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Steps')
        ax1.set_title('Episode via Steps')

        ax2.plot(np.arange(len(cost)),     cost,     'r', lw=0.8, alpha=0.8)
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Cost (Q-delta)')
        ax2.set_title('Episode via Cost')

        ax3.plot(np.arange(len(smoothed)), smoothed, 'g', lw=1.2)
        ax3.set_xlabel('Episode'); ax3.set_ylabel('Steps (smoothed)')
        ax3.set_title('Steps — Moving Average')

        plt.tight_layout()
        plt.savefig("Results/e1_qlearning_results.png", dpi=120)
        plt.close('all')
        print("  [INFO] Q-Learning plots saved → Results/e1_qlearning_results.png")


# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE 2 — Deep Q-Network
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE   = 15
STATE_SIZE  = GRID_SIZE * GRID_SIZE * 2   # 450
ACTION_SIZE = 4


class DQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(STATE_SIZE, 128)
        self.fc2  = nn.Linear(128, 64)
        self.fc3  = nn.Linear(64, ACTION_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    DQN with experience replay and hard target-network update (Episode 2).

    Parameters — same defaults as the original Episode-2 agent_brain.py.
    """

    def __init__(self, actions, learning_rate=0.001, reward_decay=0.98,
                 e_greedy=1.0, e_greedy_min=0.01, e_greedy_decay=0.995,
                 batch_size=64, memory_size=10_000, target_update_freq=100):

        self.actions            = actions
        self.lr                 = learning_rate
        self.gamma              = reward_decay
        self.epsilon            = e_greedy
        self.epsilon_min        = e_greedy_min
        self.epsilon_decay      = e_greedy_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step         = 0

        self.memory     = deque(maxlen=memory_size)
        self.online_net = DQNNetwork()
        self.target_net = DQNNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.MSELoss()

    def state_to_tensor(self, state):
        agent_pos, goal_pos = state
        one_hot  = np.zeros(STATE_SIZE, dtype=np.float32)
        one_hot[agent_pos[0] * GRID_SIZE + agent_pos[1]] = 1.0
        one_hot[GRID_SIZE * GRID_SIZE + goal_pos[0] * GRID_SIZE + goal_pos[1]] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        with torch.no_grad():
            return int(torch.argmax(self.online_net(self.state_to_tensor(state))).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t  = torch.cat([self.state_to_tensor(s) for s in states])
        ns_t = torch.cat([self.state_to_tensor(s) for s in next_states])
        a_t  = torch.LongTensor(actions)
        r_t  = torch.FloatTensor(rewards)
        d_t  = torch.FloatTensor(dones)

        q_pred   = self.online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next   = self.target_net(ns_t).max(1)[0]
        q_target = r_t + self.gamma * q_next * (1 - d_t)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Terminal display ──────────────────────────────────────────────────── #
    def print_q_table(self):
        """
        Print predicted Q-values for each cell on the best route in the
        same structured table style as Episode 2's console output.
        """
        e = final_states()
        if not e:
            print("  (No final route recorded for DQN — agent may not have reached goal)")
            return

        goal_pos     = e[max(e.keys())]   # last position in best route
        action_names = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        sep  = "=" * 72
        hdr  = "-" * 72

        print()
        print(sep)
        print("  [DQN — E2]  Q-VALUES ALONG THE BEST ROUTE  (online network)")
        print(sep)
        print(f"  States on best path : {len(e)}")
        print(hdr)
        print(f"  {'Step':>5}  {'State (row,col)':>16}  "
              f"{'UP':>9}  {'DOWN':>9}  {'RIGHT':>9}  {'LEFT':>9}  {'Best':>6}")
        print("  " + hdr)

        for step in sorted(e.keys()):
            pos    = e[step]
            state  = (pos, goal_pos)
            with torch.no_grad():
                q_vals = self.online_net(self.state_to_tensor(state))[0].numpy()
            best   = action_names[int(np.argmax(q_vals))]
            print(f"  {step:>5}  {str(pos):>16}  "
                  f"{q_vals[0]:>9.4f}  {q_vals[1]:>9.4f}  "
                  f"{q_vals[2]:>9.4f}  {q_vals[3]:>9.4f}  {best:>6}")

        print(sep)
        print()

    # ── Per-agent plots (3 subplots — original Episode-2 style) ─────────── #
    def plot_results(self, steps, rewards, losses):
        os.makedirs("Results", exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Episode 2 — DQN Training Results",
                     fontsize=13, fontweight='bold')

        ax1.plot(np.arange(len(steps)),   steps,   'b', lw=0.8, alpha=0.8)
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Steps')
        ax1.set_title('Episode via Steps')

        ax2.plot(np.arange(len(rewards)), rewards, 'r', lw=0.8, alpha=0.8)
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Total Reward')
        ax2.set_title('Episode via Reward')

        ax3.plot(np.arange(len(losses)),  losses,  'g', lw=0.8, alpha=0.8)
        ax3.set_xlabel('Episode'); ax3.set_ylabel('Loss')
        ax3.set_title('Episode via Loss')

        plt.tight_layout()
        plt.savefig("Results/e2_dqn_results.png", dpi=120)
        plt.close('all')
        print("  [INFO] DQN plots saved → Results/e2_dqn_results.png")


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON PLOTS
#  Three subplots on one figure (matching Episode-2 visual style):
#    1. Episode via Steps   — both E1 and E2 overlaid
#    2. Episode via Reward  — E1 cost (left axis) vs E2 reward (right axis)
#    3. Episode via Loss    — E2 DQN loss; E1 N/A noted in legend
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(e1_steps, e1_cost, e2_steps, e2_rewards, e2_losses,
                    e1_goals=None, e2_goals=None):
    """
    Print % accuracy comparison and generate the combined 2×3 all-metrics figure.

    Layout
    ──────
    Row 0 │ E1 Steps/ep │ E2 Steps/ep │ Steps overlay (both)
    Row 1 │ E1 Cost/ep  │ E2 Reward/ep│ E2 Loss/ep

    Parameters
    ----------
    e1_steps, e1_cost           : Q-Learning per-episode data
    e2_steps, e2_rewards, e2_losses : DQN per-episode data
    e1_goals, e2_goals          : list[bool] — True when episode reached goal
    """
    os.makedirs("Results", exist_ok=True)

    # ── Helper ───────────────────────────────────────────────────────────── #
    def smooth(data, w=30):
        w = min(w, max(1, len(data)))
        return np.convolve(data, np.ones(w) / w, mode='valid')

    # ── Accuracy % calculation ────────────────────────────────────────────── #
    e1_rate = (sum(e1_goals) / len(e1_goals) * 100) if e1_goals else 0.0
    e2_rate = (sum(e2_goals) / len(e2_goals) * 100) if e2_goals else 0.0

    # Step efficiency: lower mean steps = better convergence
    e1_mean = float(np.mean(e1_steps))
    e2_mean = float(np.mean(e2_steps))
    step_eff_e1 = (1.0 / e1_mean) if e1_mean > 0 else 0
    step_eff_e2 = (1.0 / e2_mean) if e2_mean > 0 else 0
    total_eff   = step_eff_e1 + step_eff_e2

    # DQN accuracy relative to Q-Learning  (goal-success based)
    if e1_rate > 0:
        dqn_vs_ql = e2_rate / e1_rate * 100
    else:
        dqn_vs_ql = 0.0

    # Combined score (goal-rate 70% weight + step-efficiency 30% weight)
    e1_score = 0.7 * e1_rate + 0.3 * (step_eff_e1 / total_eff * 100 if total_eff > 0 else 50)
    e2_score = 0.7 * e2_rate + 0.3 * (step_eff_e2 / total_eff * 100 if total_eff > 0 else 50)

    # ── Terminal accuracy table ───────────────────────────────────────────── #
    sep  = "=" * 68
    dash = "-" * 68
    print()
    print(sep)
    print("  ACCURACY COMPARISON — E1 (Q-Learning)  vs  E2 (DQN)")
    print(sep)
    print(f"  {'Metric':<38} {'Q-Learn (E1)':>12}  {'DQN (E2)':>10}")
    print("  " + dash)
    print(f"  {'Episodes run':<38} {len(e1_steps):>12}  {len(e2_steps):>10}")
    print(f"  {'Episodes reaching GOAL':<38} {sum(e1_goals) if e1_goals else 0:>12}  "
          f"{sum(e2_goals) if e2_goals else 0:>10}")
    print(f"  {'Goal success rate (%)':<38} {e1_rate:>11.1f}%  {e2_rate:>9.1f}%")
    print(f"  {'Min steps in any episode':<38} {min(e1_steps):>12}  {min(e2_steps):>10}")
    print(f"  {'Mean steps/episode':<38} {e1_mean:>12.1f}  {e2_mean:>10.1f}")
    print(f"  {'Std steps':<38} {np.std(e1_steps):>12.2f}  {np.std(e2_steps):>10.2f}")
    print(f"  {'Final cost / reward (last ep)':<38} {e1_cost[-1]:>12.4f}  {e2_rewards[-1]:>10.4f}")
    if e2_losses:
        print(f"  {'Final DQN loss (last ep)':<38} {'—':>12}  {e2_losses[-1]:>10.4f}")
    print("  " + dash)
    print(f"  {'Combined efficiency score':<38} {e1_score:>11.1f}%  {e2_score:>9.1f}%")
    print()
    print(f"  ★  DQN accuracy relative to Q-Learning  :  {dqn_vs_ql:>6.1f} %")
    if dqn_vs_ql >= 100:
        verdict = "DQN matches or EXCEEDS Q-Learning."
    elif dqn_vs_ql >= 70:
        verdict = "DQN is competitive with Q-Learning."
    elif dqn_vs_ql >= 40:
        verdict = "DQN is learning but needs more episodes."
    else:
        verdict = "DQN has not converged — increase EPISODES_DQN."
    print(f"  ➤  Verdict : {verdict}")
    print(sep)
    print()

    # ── Combined 2 × 3 figure ─────────────────────────────────────────────── #
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("E1 (Q-Learning)  vs  E2 (DQN) — All Training Metrics",
                 fontsize=14, fontweight='bold')

    ep1 = np.arange(len(e1_steps))
    ep2 = np.arange(len(e2_steps))
    BLUE   = 'royalblue'
    ORANGE = 'tomato'

    # ── [0,0]  E1 Steps per episode ──────────────────────────────────────── #
    axes[0, 0].plot(ep1, e1_steps, color=BLUE, lw=0.5, alpha=0.4)
    axes[0, 0].plot(np.arange(len(smooth(e1_steps))), smooth(e1_steps),
                    color=BLUE, lw=2)
    axes[0, 0].set_title('E1 — Episode via Steps')
    axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Steps')

    # ── [0,1]  E2 Steps per episode ──────────────────────────────────────── #
    axes[0, 1].plot(ep2, e2_steps, color=ORANGE, lw=0.5, alpha=0.4)
    axes[0, 1].plot(np.arange(len(smooth(e2_steps))), smooth(e2_steps),
                    color=ORANGE, lw=2)
    axes[0, 1].set_title('E2 — Episode via Steps')
    axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Steps')

    # ── [0,2]  Steps overlay — both agents ───────────────────────────────── #
    axes[0, 2].plot(np.arange(len(smooth(e1_steps))), smooth(e1_steps),
                    color=BLUE,   lw=2, label=f'Q-Learning (E1)  μ={e1_mean:.0f}')
    axes[0, 2].plot(np.arange(len(smooth(e2_steps))), smooth(e2_steps),
                    color=ORANGE, lw=2, label=f'DQN (E2)  μ={e2_mean:.0f}')
    axes[0, 2].set_title('Steps Comparison (smoothed)')
    axes[0, 2].set_xlabel('Episode'); axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend(fontsize=9)

    # ── [1,0]  E1 Cost per episode ────────────────────────────────────────── #
    axes[1, 0].plot(ep1, e1_cost, color=BLUE, lw=0.5, alpha=0.4)
    axes[1, 0].plot(np.arange(len(smooth(e1_cost))), smooth(e1_cost),
                    color=BLUE, lw=2)
    axes[1, 0].set_title('E1 — Episode via Cost (Q-delta)')
    axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('Cost')

    # ── [1,1]  E2 Reward per episode ─────────────────────────────────────── #
    axes[1, 1].plot(ep2, e2_rewards, color=ORANGE, lw=0.5, alpha=0.4)
    axes[1, 1].plot(np.arange(len(smooth(e2_rewards))), smooth(e2_rewards),
                    color=ORANGE, lw=2)
    axes[1, 1].set_title('E2 — Episode via Reward')
    axes[1, 1].set_xlabel('Episode'); axes[1, 1].set_ylabel('Total Reward')

    # ── [1,2]  E2 Loss per episode ────────────────────────────────────────── #
    axes[1, 2].plot(ep2, e2_losses, color='seagreen', lw=0.5, alpha=0.4)
    axes[1, 2].plot(np.arange(len(smooth(e2_losses))), smooth(e2_losses),
                    color='seagreen', lw=2, label=f'DQN Loss  ({e2_rate:.1f}% goal rate)')
    axes[1, 2].axhline(0, color=BLUE, lw=1, linestyle='--',
                       label=f'Q-Learning  ({e1_rate:.1f}% goal rate)')
    axes[1, 2].set_title('E2 — Episode via Loss')
    axes[1, 2].set_xlabel('Episode'); axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend(fontsize=8)

    # Accuracy annotation at bottom
    fig.text(0.5, 0.01,
             f"DQN accuracy vs Q-Learning: {dqn_vs_ql:.1f}%  |  "
             f"Goal success — Q-Learning: {e1_rate:.1f}%   DQN: {e2_rate:.1f}%  |  "
             f"{verdict}",
             ha='center', fontsize=10,
             color='darkred' if dqn_vs_ql < 70 else 'darkgreen',
             fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = "Results/comparison_all_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"  [INFO] Combined metrics figure saved → {out}")

    # Open all result images via the OS default viewer
    for fname in ["Results/e1_qlearning_results.png",
                  "Results/e2_dqn_results.png",
                  out]:
        try:
            os.startfile(os.path.abspath(fname))
        except Exception:
            pass   # non-Windows: open files manually


