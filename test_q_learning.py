# import argparse
# import gymnasium as gym
# import
# from q_learning_cartpole import QLearningAgent, NoisyObservationWrapper

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Test Q-Learning for CartPole with Sensor Noise')
#     parser.add_argument('--model', type=str, required=True, help='Model filename to load')
#     parser.add_argument('--tests', type=int, default=200, help='Number of tests to conduct')
#     args = parser.parse_args()

#     test_env = gym.make('CartPole-v1', render_mode='human')
#     noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
#     agent = QLearningAgent(noisy_test_env, model_filename=args.model)

#     def test(episodes=200):
#         total_rewards = []

#         for episode in range(episodes):
#             current_state, info = agent.env.reset()
#             current_state = agent.discretize(current_state)
#             done = False
#             total_reward = 0

#             while not done:
#                 if episode % 20 == 0:
#                     agent.env.render()
#                 action = np.argmax(agent.q_table[current_state])  # Choose the action with the highest Q-value
#                 next_state, reward, done, _, _ = agent.env.step(action)
#                 next_state = agent.discretize(next_state)
#                 current_state = next_state
#                 total_reward += reward

#             total_rewards.append(total_reward)
#             print("Test Episode: {}, Total reward: {}".format(episode, total_reward))

#         average_reward = np.mean(total_rewards)
#         highest_reward = np.max(total_rewards)
#         lowest_reward = np.min(total_rewards)

#         print("Average Reward: {}".format(average_reward))
#         print("Highest Reward: {}".format(highest_reward))
#         print("Lowest Reward: {}".format(lowest_reward))

#         agent.env.close()

#     test(args.tests)


import gymnasium as gym
import numpy as np
import math
import pickle
import argparse

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2,  # cart position noise range is -2 to 2
            0.5,  # cart velocity noise range is -0.5 to 0.5
            math.radians(20),  # pole angle noise range is -20 degrees to 20 degrees
            math.radians(0.5)  # pole angular velocity noise range is -0.5 degrees/s to 0.5 degrees/s
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs

class QLearningAgent:
    def __init__(self, env, bins=(15, 15, 15, 15), alpha=0.1, gamma=0.7, epsilon=0.7, epsilon_decay=0.999, epsilon_min=0.01, model_filename=None):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros(self.bins + (env.action_space.n,))
        
        # Define smaller bin limits compared to the terminal states
        self.bins_limits = [
            (-4.8, 4.8),  # cart position (terminal state is -4.8 to 4.8)
            (-10, 10),  # cart velocity (terminal state is -inf to inf, but we use a practical range)
            (-math.radians(24), math.radians(24)),  # pole angle (terminal state is -math.radians(24) to math.radians(24))
            (-math.radians(10), math.radians(10))  # pole angular velocity (terminal state is -inf to inf, but we use a practical range)
        ]

    def discretize(self, obs):
        ratios = [(obs[i] - self.bins_limits[i][0]) / (self.bins_limits[i][1] - self.bins_limits[i][0]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filename}")

    def test(self, model_filename, episodes=1):
        self.load_model(model_filename)
        for episode in range(episodes):
            current_state, info = self.env.reset()
            current_state = self.discretize(current_state)
            done = False
            total_reward = 0

            while not done:
                self.env.render()
                action = np.argmax(self.q_table[current_state])
                next_state, reward, done, _, _ = self.env.step(action)
                cart_position, cart_velocity, pole_angle, pole_angular_velocity = next_state
                if abs(cart_position) < 0.1:
                    reward += 1  # Reward for being close to the middle
                if abs(pole_angle) < np.radians(5):
                    reward += 1  # Reward for being close to vertical
                next_state = self.discretize(next_state)
                current_state = next_state
                total_reward += reward

            print(f"Test Episode: {episode}, Total reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Q-Learning for CartPole with Sensor Noise')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file to load')
    parser.add_argument('--episodes', type=int, default=1, help='Number of test episodes')
    args = parser.parse_args()

    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent = QLearningAgent(noisy_test_env)
    agent.test(model_filename=args.model, episodes=args.episodes)
    test_env.close()
