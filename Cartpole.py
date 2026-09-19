import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1",render_mode = 'human')
n_bins = (6, 12)

lower_bouns = [env.observation_space.low[2], - 3.5]
upper_bouns = [env.observation_space.high[2], 3.5]

def discretize(obs):
    cart_velocity, pole_angle = obs[2], obs[3]
    ratios = [(cart_velocity - low) / (high - low) for obs, low, high in zip([cart_velocity, pole_angle], lower_bouns, upper_bouns)]
    new_obs = [int(round((n_bins[i] - 1 ) * ratios[i])) for i in range(len(n_bins))]
    new_obs = [min(n_bins[i] - 1,max(0, new_obs[i])) for i in range(len(n_bins))]
    return tuple(new_obs)

q_table = np.zeros(n_bins    + (env.action_space.n,))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.955
min_epsilon = 0.1
episodes = 1000
rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    total_reward = 0

    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        best_next_action =  np.max(q_table[next_state])
        q_table[state + (action,)] += alpha * (reward + gamma * best_next_action - q_table[state + (action,)])
        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

    if epsilon % 500 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")



plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning in cartpole")
plt.grid(True)
plt.show()

env.close()        

