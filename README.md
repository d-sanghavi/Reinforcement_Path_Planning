# Reinforcement Learning Based Path Planning (RPP)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Reinforcement Learning](https://img.shields.io/badge/Algorithm-Q--Learning-green)
![Environment](https://img.shields.io/badge/Environment-GridWorld-orange)
![Status](https://img.shields.io/badge/Project-Phase--1-yellow)

A **Reinforcement Learning based autonomous path planning system** where an agent learns to navigate through a **grid-based urban environment** while avoiding obstacles and minimizing travel cost.

This project is developed as part of the **IPD (Interdisciplinary Project Development)** course.

The agent learns the **optimal path to the goal using Q-Learning**, interacting with the environment over multiple episodes.

---

# Project Overview

Autonomous navigation is a key component in **robotics, smart transportation, and AI systems**.

This project simulates an **urban grid world** containing:

- Buildings
- Traffic signals
- Trees
- Shops
- Pedestrian crossings
- Road barriers
- Obstacles

The Reinforcement Learning agent learns to:

- Explore the environment
- Avoid obstacles
- Minimize path cost
- Reach the destination efficiently

---

# Environment Description

The environment is represented as a **2D Grid World**.

Each cell may represent:

| Symbol | Meaning |
|------|------|
| Empty Cell | Road |
| Tree | Obstacle |
| Building | Restricted zone |
| Traffic Signal | Movement constraint |
| Shop / Structure | Static obstacle |
| Goal | Target destination |

The agent moves through the grid while learning the **optimal policy**.

---

# Reinforcement Learning Algorithm

This project implements **Q-Learning**, a model-free reinforcement learning algorithm.

### Q-Learning Update Rule
Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]


Where:

| Parameter | Meaning |
|--------|--------|
| s | Current state |
| a | Action taken |
| r | Reward received |
| s' | Next state |
| α | Learning rate |
| γ | Discount factor |

---

# State Space

Each state corresponds to the **agent position in the grid**.

Example:
(state_x, state_y)


---

# Action Space

The agent can perform **four actions**:
UP
DOWN
LEFT
RIGHT


---

# Reward System

| Event | Reward |
|------|------|
| Move to valid cell | Small negative reward |
| Hit obstacle | Large negative reward |
| Reach goal | Positive reward |
| Invalid move | Penalty |

This encourages the agent to **reach the goal with minimal steps**.

---

# Project Structure
Reinforcement_Path_Planning
│
├── agent_brain.py # Q-learning agent implementation
├── env.py # Grid world environment
├── run_agent.py # Training and execution script
│
├── outputs
│ ├── environment.png
│ └── training_graph.png
│
├── README.md
└── requirements.txt


---

# Installation

Clone the repository

```bash
git clone https://github.com/d-sanghavi/Reinforcement_Path_Planning.git
cd Reinforcement_Path_Planning
```
Create virtual environment
```bash
python -m venv rl_env
```
Activate environment
Windows
```bash
rl_env\Scripts\activate
```
Linux / Mac
```bash
source rl_env/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is not available:
```bash
pip install numpy matplotlib pygame
```
Running the Project

To train the RL agent:
```bash
python run_agent.py
```
The script will:

Initialize the environment

Train the agent using Q-Learning

Visualize the grid world

Plot training performance graphs

Phase 1 Results
Environment Visualization

Example: [RESULT](RL_Q-Learning_E1\Results\staticQLearning.png)

Training Performance

Add the training graph image like this:

Exanple: [Training Graph](RL_Q-Learning_E1\Results\Figure_1.png)

Author
Suhani Gupta (Leader)
Paridhi Jain
Dhruv Sanghavi
Khush Shah
IPD Project – Reinforcement Learning Path Planning for (atmost 2) robots 

License

This project is licensed under the MIT License.
