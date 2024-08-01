# test_dqn.py

import argparse
import torch
import gymnasium as gym
from clean_DQN import DQNAgent, NoisyObservationWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN for CartPole with Sensor Noise')
    parser.add_argument('--model', type=str, help='Model filename to load', required=True)
    args = parser.parse_args()

    # test_env = gym.make('CartPole-v1')
    test_env = gym.make('CartPole-v1')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent = DQNAgent(noisy_test_env)
    agent.load_model(args.model)
    agent.test(model_filename=args.model, episodes=10)
    test_env.close()


# # test_dqn_cartpole.py

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import argparse 

# class NoisyObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, noise_std=0.1):
#         super(NoisyObservationWrapper, self).__init__(env)
#         self.noise_std = noise_std

#     def observation(self, obs):
#         noise = np.random.normal(0, self.noise_std, size=obs.shape)
#         noisy_obs = obs + noise
#         return noisy_obs

# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 128)
#         self.fc5 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

# class DQNAgent:
#     def __init__(self, env, model_filename=None):
#         self.env = env
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
#         if model_filename:
#             self.load_model(model_filename)

#     def load_model(self, filename):
#         self.model.load_state_dict(torch.load(filename, map_location=self.device))
#         self.model.eval()
#         print(f"Model loaded from {filename}")

#     def test(self, model_filename, episodes=1):
#         self.load_model(model_filename)
#         for episode in range(episodes):
#             state, info = self.env.reset()
#             total_reward = 0
#             done = False

#             while not done:
#                 self.env.render()
#                 state = torch.FloatTensor(state).to(self.device)
#                 with torch.no_grad():
#                     action = torch.argmax(self.model(state)).item()
#                 next_state, reward, done, _, _ = self.env.step(action)

#                 # Custom reward function
#                 cart_position, cart_velocity, pole_angle, pole_velocity = next_state
#                 reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)
                
#                 if done and total_reward < 500:
#                     reward = -1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)  # Penalize if the episode ends prematurely


#                 state = next_state
#                 total_reward += reward

#             print(f"Test Episode: {episode}, Total reward: {total_reward}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Test DQN for CartPole with Sensor Noise')
#     parser.add_argument('--model', type=str, required=True, help='Path to the model file to load')
#     parser.add_argument('--episodes', type=int, default=1, help='Number of test episodes')
#     args = parser.parse_args()

#     test_env = gym.make('CartPole-v1', render_mode='human')
#     noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
#     agent = DQNAgent(noisy_test_env)
#     agent.test(model_filename=args.model, episodes=args.episodes)
#     test_env.close()
