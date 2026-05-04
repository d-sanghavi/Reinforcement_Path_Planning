# run_agent.py — E1 vs E2 Accuracy Comparison — main entry point
#
# Full pipeline
# ─────────────
#  Phase 1  Placement   User places obstacles / START / GOAL in pygame window.
#  Phase 2  Q-Learning  Episode-1 algorithm runs 1 000 episodes on the grid.
#                       → Best route stored as  ql_route
#                       → Q-table printed to terminal (Episode-2 style)
#  Phase 3  DQN         Episode-2 algorithm runs 1 000 episodes on same grid.
#                       → Best route stored as  dqn_route
#                       → Q-value table printed to terminal (Episode-2 style)
#  Phase 4  Dual route  Both routes drawn on one pygame window:
#                           BLUE   = Q-Learning path
#                           ORANGE = DQN path
#  Phase 5  Plots       Combined 2×3 figure with all metrics + accuracy summary.
#
# Usage:
#   cd E1_vs_E2_accuracy_comparison
#   python run_agent.py

from env         import Environment
from agent_brain import QLearningTable, DQNAgent, plot_comparison
import time

# ── Hyper-parameters ──────────────────────────────────────────────────────── #
EPISODES_QL  = 1000
EPISODES_DQN = 1000
MAX_STEPS    = 600     # Hard per-episode step cap (both agents)
GOAL_REWARD  = 9.0     # Threshold: reward >= this → episode ended at goal


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Q-Learning (Episode 1 algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def run_qlearning(env):
    """
    Train the tabular Q-Learning agent on the already-configured grid.

    Returns
    -------
    steps_list  : list[int]
    cost_list   : list[float]
    goal_flags  : list[bool]   — True when the episode ended at the goal
    ql_route    : dict{step: (row,col)}
    RL          : trained QLearningTable
    """
    print()
    print("=" * 72)
    print("  PHASE 2 — Q-LEARNING  (Episode 1 Algorithm)")
    print("=" * 72)

    RL = QLearningTable(actions=list(range(env.n_actions)))

    steps_list = []
    cost_list  = []
    goal_flags = []          # True = reached goal this episode

    for episode in range(1, EPISODES_QL + 1):
        observation  = env.reset()
        i            = 0
        cost         = 0.0
        reached_goal = False

        while True:
            # Render every 20th episode so the window stays responsive
            if episode % 20 == 0:
                env.render(episode=episode, total=EPISODES_QL, epsilon=RL.epsilon)
            else:
                env.pump()

            action             = RL.choose_action(str(observation))
            obs_, reward, done = env.step(action)
            cost              += RL.learn(str(observation), action,
                                          reward, str(obs_))
            observation        = obs_
            i                 += 1

            if reward >= GOAL_REWARD:
                reached_goal = True

            if done or i >= MAX_STEPS:
                steps_list.append(i)
                cost_list.append(cost)
                goal_flags.append(reached_goal)
                break

        if episode % 100 == 0 or episode == 1:
            print(f"  [Q-Learning]  Episode {episode:>5}/{EPISODES_QL}"
                  f"  |  Steps: {i:>4}"
                  f"  |  Cost: {cost:>9.4f}"
                  f"  |  ε: {RL.epsilon:.3f}"
                  f"  |  Goal: {'YES' if reached_goal else 'no'}")

    ql_route = dict(env.best_route)

    print()
    print(f"  Q-Learning complete.")
    print(f"  Shortest route : {env.shortest} steps  |  Longest: {env.longest} steps")
    rate = sum(goal_flags) / len(goal_flags) * 100
    print(f"  Goal success   : {sum(goal_flags)}/{len(goal_flags)} episodes  ({rate:.1f}%)")

    # Show best route (BLUE, 2 s) then update final_states() for print_q_table
    env.update_global_route(color=(50, 100, 220), title="-- Q-LEARNING ROUTE --")

    # Print Q-tables to terminal
    RL.print_q_table()

    # Save per-agent plot (no plt.show — uses Agg backend)
    RL.plot_results(steps_list, cost_list)

    return steps_list, cost_list, goal_flags, ql_route, RL


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — DQN (Episode 2 algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def run_dqn(env):
    """
    Train the DQN agent on the same grid (obstacles / start / goal preserved).
    Route tracking is reset so DQN discovers its own best path independently.

    Returns
    -------
    steps_list   : list[int]
    reward_list  : list[float]
    loss_list    : list[float]
    goal_flags   : list[bool]
    dqn_route    : dict{step: (row,col)}
    agent        : trained DQNAgent
    """
    print()
    print("=" * 72)
    print("  PHASE 3 — DQN  (Episode 2 Algorithm)")
    print("=" * 72)

    # Fresh route tracking for DQN
    env.best_route = {}
    env.first_goal = True
    env.shortest   = 0
    env.longest    = 0

    agent = DQNAgent(actions=list(range(4)))

    steps_list  = []
    reward_list = []
    loss_list   = []
    goal_flags  = []          # True = reached goal this episode

    for episode in range(1, EPISODES_DQN + 1):
        state        = (env.reset(), env.goal_pos)
        total_reward = 0.0
        total_loss   = 0.0
        loss_count   = 0
        i            = 0
        reached_goal = False

        while True:
            # Render every 20th episode so the agent moves visibly on screen;
            # just pump events for the 19 episodes in between (much faster).
            if episode % 20 == 0:
                env.render(episode=episode, total=EPISODES_DQN,
                           epsilon=agent.epsilon)
            else:
                env.pump()

            action                 = agent.choose_action(state)
            next_pos, reward, done = env.step(action)
            next_state             = (next_pos, env.goal_pos)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss > 0:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            state         = next_state
            i            += 1

            if reward >= GOAL_REWARD:
                reached_goal = True

            if done or i >= MAX_STEPS:
                steps_list.append(i)
                reward_list.append(total_reward)
                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
                loss_list.append(avg_loss)
                goal_flags.append(reached_goal)

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                if episode % 100 == 0 or episode == 1:
                    print(f"  [DQN]  Episode {episode:>5}/{EPISODES_DQN}"
                          f"  |  Steps: {i:>4}"
                          f"  |  Reward: {total_reward:>8.2f}"
                          f"  |  Loss: {avg_loss:>8.4f}"
                          f"  |  ε: {agent.epsilon:.3f}"
                          f"  |  Goal: {'YES' if reached_goal else 'no'}")
                    time.sleep(0.4)   # brief pause so user can read the log line
                break

    dqn_route = dict(env.best_route)

    print()
    print(f"  DQN complete.")
    print(f"  Shortest route : {env.shortest} steps  |  Longest: {env.longest} steps")
    rate = sum(goal_flags) / len(goal_flags) * 100
    print(f"  Goal success   : {sum(goal_flags)}/{len(goal_flags)} episodes  ({rate:.1f}%)")

    # Show best route (ORANGE, 2 s) then update final_states()
    env.update_global_route(color=(255, 140, 0), title="-- DQN ROUTE --")

    # Print Q-value table
    agent.print_q_table()

    # Save per-agent plot (no plt.show — uses Agg backend)
    agent.plot_results(steps_list, reward_list, loss_list)

    return steps_list, reward_list, loss_list, goal_flags, dqn_route, agent


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       E1  vs  E2 — Reinforcement Learning Accuracy Comparison        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Phase 1 > Place obstacles, START and GOAL — then click DONE         ║")
    print("║  Phase 2 > Q-Learning (E1) trains on the grid                        ║")
    print("║  Phase 3 > DQN (E2) trains on the same grid  (10x faster render)     ║")
    print("║  Phase 4 > Both paths drawn on one window (Blue=QL | Orange=DQN)     ║")
    print("║  Phase 5 > Combined 2x3 metrics figure + accuracy % printed          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── Build environment ────────────────────────────────────────────────── #
    env = Environment()

    # ── Phase 1 — obstacle placement ────────────────────────────────────── #
    print("  [INFO] Pygame window open — place obstacles then click DONE.")
    env.run_placement_phase()
    print("  [INFO] Grid locked. Starting training...\n")

    # ── Phase 2 — Q-Learning ────────────────────────────────────────────── #
    e1_steps, e1_cost, e1_goals, ql_route, rl_agent = run_qlearning(env)

    # ── Phase 3 — DQN ───────────────────────────────────────────────────── #
    e2_steps, e2_rewards, e2_losses, e2_goals, dqn_route, dqn_agent = run_dqn(env)

    # ── Phase 4 — dual-route visualisation ──────────────────────────────── #
    print()
    print("=" * 72)
    print("  PHASE 4 — DUAL ROUTE VISUALISATION")
    print("  BLUE   circles = Q-Learning best path")
    print("  ORANGE circles = DQN best path")
    print("  Close the pygame window to proceed to comparison plots.")
    print("=" * 72)
    env.draw_dual_routes(ql_route, dqn_route)

    # ── Phase 5 — combined metrics + accuracy % ──────────────────────────── #
    print()
    print("=" * 72)
    print("  PHASE 5 — COMPARISON PLOTS & ACCURACY SUMMARY")
    print("=" * 72)
    plot_comparison(e1_steps, e1_cost, e2_steps, e2_rewards, e2_losses,
                    e1_goals, e2_goals)

    print("  [DONE]  Results saved to  Results/")
    print("          •  Results/e1_qlearning_results.png")
    print("          •  Results/e2_dqn_results.png")
    print("          •  Results/comparison_all_metrics.png")
    print()
