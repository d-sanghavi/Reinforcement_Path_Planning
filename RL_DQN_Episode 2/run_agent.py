from env import Environment
from agent_brain import DQNAgent

EPISODES  = 1000
MAX_STEPS = 600

def update(env, agent):
    steps_per_episode   = []
    rewards_per_episode = []
    losses_per_episode  = []

    for episode in range(1, EPISODES + 1):
        state        = (env.reset(), env.goal_pos)
        total_reward = 0
        total_loss   = 0
        loss_count   = 0
        i            = 0

        while True:
            # Render every 10 episodes only
            if episode % 10 == 0:
                env.render(episode=episode, total=EPISODES, epsilon=agent.epsilon)
            else:
                env.pump()

            # Choose action
            action = agent.choose_action(state)

            # Step
            next_state, reward, done = env.step(action)
            next_state = (next_state, env.goal_pos)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Learn
            loss = agent.learn()
            if loss > 0:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            state         = next_state
            i            += 1

            if done or i >= MAX_STEPS:
                steps_per_episode.append(i)
                rewards_per_episode.append(total_reward)
                avg_loss = total_loss / loss_count if loss_count > 0 else 0
                losses_per_episode.append(avg_loss)

                # Epsilon decay once per episode
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                print(f"Episode {episode}/{EPISODES} | "
                      f"Steps: {i} | "
                      f"Reward: {total_reward:.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
                break

    # Plot results (do this before env.final() because env.final() loops forever until exit)
    agent.plot_results(steps_per_episode, rewards_per_episode, losses_per_episode)

    # Show final route
    env.final()


if __name__ == "__main__":
    env   = Environment()

    # Phase 1: user places obstacles
    env.run_placement_phase()

    # Phase 2: DQN training
    agent = DQNAgent(actions=list(range(4)))
    update(env, agent)