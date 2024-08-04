# dqn_cartpole.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import datetime
import argparse
import matplotlib.pyplot as plt
import math
import json
import os
import csv

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2,  # cart position noise range is -2 to 2
            0.5,  # cart velocity noise range is -0.5 to 0.5
            math.radians(2),  # pole angle noise range is -20 degrees to 20 degrees
            math.radians(0.5)  # pole angular velocity noise range is -0.5 degrees/s to 0.5 degrees/s
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs

# A deeper model for better performance
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def reward_fun1(cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
    reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
    if done and total_reward < 500:
        reward = -0.1 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
    return reward

def reward_fun2(cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
    if abs(cart_position) < 0.5 and abs(pole_angle) < 0.05:
        reward = 1.0
    else:
        reward = - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)
    if done and total_reward < 500:
        reward = -1.0
    return reward

class HyperParameters:
    def __init__(self, args):
        self.update_rate = args.update_rate
        self.gamma = args.gamma
        self.episodes = args.episodes
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.memory_size = args.memory_size
        # self.reward_fun = args.reward_fun
        self.noise_std = args.noise_std

class DQNAgent:
    def __init__(self, env, hyper_params):
        self.env = env
        self.episodes = hyper_params.episodes
        self.gamma = hyper_params.gamma
        self.epsilon = hyper_params.epsilon
        self.epsilon_min = hyper_params.epsilon_min
        self.epsilon_decay = hyper_params.epsilon_decay
        self.lr = hyper_params.lr
        self.update_rate = hyper_params.update_rate
        self.batch_size = hyper_params.batch_size
        self.memory = deque(maxlen=hyper_params.memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # self.reward_fun = globals()[hyper_params.reward_fun]

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, episodes=5000, save_filename=None):
        if save_filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f'dqn_models/dqn_{current_time}.pth'

        rewards = []
        recent_rewards = deque(maxlen=100)
        best_total_reward = -float('inf')

        os.makedirs('dqn_models', exist_ok=True)

        csv_filename = save_filename.replace('.pth', '_rewards.csv')

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Total Reward"])

            for episode in range(episodes):
                state, info = self.env.reset()
                total_reward = 0
                step = 0
                done = False

                while not done:
                    action = self.act(state)
                    next_state, reward, done, _, _ = self.env.step(action)

                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    step += 1
                    self.replay()

                    if step > 500:
                        done = True
                        print("SUCCESS!")

                rewards.append(total_reward)
                recent_rewards.append(total_reward)

                writer.writerow([episode, total_reward])

                if total_reward > best_total_reward:
                    best_total_reward = total_reward
                    self.save_model(save_filename)
                    print(f"New best total reward: {best_total_reward} - Model saved")

                if episode % self.update_rate == 0:
                    self.update_target_model()
                    print(f"Episode: {episode}, Average reward: {np.mean(rewards[-10:])}, Epsilon: {self.epsilon}, Step: {step}")

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs. Total Reward')
        plt.savefig(save_filename.replace('.pth', '_plot.png'))
        plt.show()


    def test(self, model_filename, episodes=10, disturbance_step=100, disturbance_magnitude=1.0):
        self.load_model(model_filename)
        rewards = []
        successes = 0

        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done:
                self.env.render()
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action = np.argmax(self.model(state).cpu().data.numpy())
                next_state, reward, done, _, _ = self.env.step(action)
                
                # if step == disturbance_step:
                #     next_state[3] += disturbance_magnitude

                # cart_position, cart_velocity, pole_angle, pole_velocity = next_state
                # reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
                
                # if done and step < 500:
                #     reward = -1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)

                state = next_state
                total_reward += reward
                step += 1
                if step >= 500:
                    done = True
                    successes += 1
                    print("SUCCESS!")

            rewards.append(total_reward)
            print(f"Test Episode: {episode}, Total reward: {total_reward}")

        success_rate = (successes / episodes) * 100
        print(f"Success rate: {success_rate}%")

        plt.plot(range(episodes), rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Episodes')
        plt.show()


    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN for CartPole with Sensor Noise')
    parser.add_argument('--load', type=str, help='Model filename to load', default=None)
    parser.add_argument('--save', type=str, help='Model filename to save', default=None)
    parser.add_argument('--reward', type=str, help='Reward function to use', default='reward_fun1')
    parser.add_argument('--update_rate', type=int, default=10, help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.99, help='Initial epsilon')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Epsilon decay factor')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for replay')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay memory size')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Standard deviation of the noise')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes for training')
    args = parser.parse_args()

    hyper_params = HyperParameters(args)

    save_filename = args.save
    if not save_filename:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f'dqn_models/dqn_{current_time}.pth'

    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=hyper_params.noise_std)
    agent = DQNAgent(noisy_train_env, hyper_params)
    if args.load:
        agent.load_model(args.load)
    agent.train(episodes=agent.episodes, save_filename=save_filename)

    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=hyper_params.noise_std)
    agent.env = noisy_test_env
    agent.test(model_filename=save_filename, episodes=10)
    test_env.close()




# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# import datetime
# import argparse
# import matplotlib.pyplot as plt
# import math
# import csv

# class NoisyObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, noise_std=0.3):
#         super(NoisyObservationWrapper, self).__init__(env)
#         self.noise_std = noise_std
#         self.observation_space = env.observation_space
        
#         # Define the range for each observation component
#         self.obs_ranges = [
#             2,  # cart position noise range is -2 to 2
#             0.5,  # cart velocity noise range is -0.5 to 0.5
#             math.radians(2),  # pole angle noise range is -2 degrees to 2 degrees
#             math.radians(0.5)  # pole angular velocity noise range is -0.5 degrees/s to 0.5 degrees/s
#         ]

#     def observation(self, obs):
#         noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
#         noisy_obs = obs + noise
#         return noisy_obs

# # A deeper model for better performance
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

# # Define the hyperparameters class
# class HyperParameters:
#     def __init__(self, update_rate=10, gamma=0.9, epsilon=0.99, epsilon_min=0.01, epsilon_decay=0.9995, 
#                  lr=0.0001, batch_size=64, memory_size=10000, reward_fun='reward_fun1', 
#                  episodes=100000, noise_std=0.1, test_episodes=100):
#         self.update_rate = update_rate
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.lr = lr
#         self.batch_size = batch_size
#         self.memory_size = memory_size
#         self.reward_fun = reward_fun
#         self.episodes = episodes
#         self.noise_std = noise_std
#         self.test_episodes = test_episodes

# class DQNAgent:
#     def __init__(self, hyperparams):
#         self.env = NoisyObservationWrapper(gym.make('CartPole-v1'), noise_std=hyperparams.noise_std)
#         self.gamma = hyperparams.gamma
#         self.epsilon = hyperparams.epsilon
#         self.epsilon_min = hyperparams.epsilon_min
#         self.epsilon_decay = hyperparams.epsilon_decay
#         self.lr = hyperparams.lr
#         self.update_rate = hyperparams.update_rate
#         self.batch_size = hyperparams.batch_size
#         self.memory = deque(maxlen=hyperparams.memory_size)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.model = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
#         self.target_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
#         self.update_target_model()

#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.loss_fn = nn.MSELoss()

#         self.reward_fun = getattr(self, hyperparams.reward_fun)

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return self.env.action_space.sample()
#         state = torch.FloatTensor(state).to(self.device)
#         with torch.no_grad():
#             q_values = self.model(state)
#         return np.argmax(q_values.cpu().data.numpy())

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         minibatch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*minibatch)
        
#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_q_values = self.target_model(next_states).max(1)[0]
#         target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

#         loss = self.loss_fn(q_values, target_q_values.detach())
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def reward_fun1(self, cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
#         reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
#         if done and total_reward < 500:
#             reward = -0.1 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
#         return reward

#     def reward_fun2(self, cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
#         if abs(cart_position) < 0.5 and abs(pole_angle) < 0.05:
#             reward = 1.0
#         else:
#             reward = - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)
#         if done and total_reward < 500:
#             reward = -1.0
#         return reward

#     def train(self, postfix, episodes):
#         rewards = []
#         best_total_reward = -float('inf')

#         for episode in range(episodes):
#             state, info = self.env.reset()
#             total_reward = 0
#             step = 0
#             done = False

#             while not done:
#                 action = self.act(state)
#                 next_state, reward, done, _, _ = self.env.step(action)

#                 # reward = self.reward_fun(*next_state, total_reward, done)

#                 self.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 total_reward += reward
#                 step += 1
#                 self.replay()

#             rewards.append(total_reward)

#             if total_reward > best_total_reward:
#                 best_total_reward = total_reward
#                 self.save_model(f"highest_reward_{postfix}.pth")
#                 print(f"New best total reward: {best_total_reward} - Model saved")

#             if episode % self.update_rate == 0:
#                 self.update_target_model()
#                 print(f"Episode: {episode}, Average reward: {np.mean(rewards[-10:])}, Epsilon: {self.epsilon}, Step: {step}")

#             if step > 500: # If the episode is successful, then stop this episode's training 
#                 print("SUCCESS!")
#                 break

#             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#         # Save rewards to CSV
#         rewards_csv_filepath = f"rewards_{postfix}.csv"
#         with open(rewards_csv_filepath, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Episode', 'Total Reward'])
#             for i, reward in enumerate(rewards):
#                 writer.writerow([i + 1, reward])
#         print(f"Rewards saved to {rewards_csv_filepath}")

#         # Save plot
#         plt.plot(rewards)
#         plt.xlabel('Episode')
#         plt.ylabel('Total Reward')
#         plt.title('Episode vs. Total Reward')
#         plt.savefig(f"dqn_rewards_vs_episodes_{postfix}.png")
#         plt.show()

#     def test(self, model_filename, episodes=100):
#         self.load_model(model_filename)
#         rewards = []
#         total_success = 0

#         for episode in range(episodes):
#             state, info = self.env.reset()
#             total_reward = 0
#             done = False
#             step = 0

#             while not done:
#                 state = torch.FloatTensor(state).to(self.device)
#                 with torch.no_grad():
#                     action = np.argmax(self.model(state).cpu().data.numpy())
#                 next_state, reward, done, _, _ = self.env.step(action)

#                 # reward = self.reward_fun(*next_state, total_reward, done)

#                 state = next_state
#                 total_reward += reward
#                 step += 1
#                 if step == 500:
#                     print("SUCCESS!")

#             rewards.append(total_reward)
#             if total_reward >= self.env.spec.reward_threshold:
#                 total_success += 1
            
#             print(f"Test Episode: {episode}, Total reward: {total_reward}")

#         accuracy = (total_success / episodes) * 100
#         print(f'Accuracy over {episodes} episodes: {accuracy:.2f}%')

#         # Plot episodes vs rewards
#         plt.plot(range(1, episodes + 1), rewards)
#         plt.xlabel('Episode')
#         plt.ylabel('Total Reward')
#         plt.title('Episodes vs Total Rewards')
#         plt.savefig(f"dqn_rewards_vs_episodes_{postfix}_test.png")
#         plt.show()

#     def save_model(self, filename):
#         torch.save(self.model.state_dict(), filename)
#         print(f"Model saved to {filename}")

#     def load_model(self, filename):
#         self.model.load_state_dict(torch.load(filename))
#         self.model.eval()
#         print(f"Model loaded from {filename}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='DQN for CartPole with Sensor Noise')
#     parser.add_argument('--load', type=str, help='Model filename to load', default=None)
#     parser.add_argument('--save', type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), help='Postfix for saved items')
#     parser.add_argument('--reward', type=str, help='Reward function to use', default='reward_fun1')
#     parser.add_argument('--test', action='store_true', help='Test the model')
#     parser.add_argument('--episodes', type=int, help='Number of episodes for training', default=50000)
#     parser.add_argument('--test_episodes', type=int, help='Number of episodes for testing', default=100)
#     parser.add_argument('--update_rate', type=int, help='Update rate for target network', default=10)
#     parser.add_argument('--gamma', type=float, help='Discount factor', default=0.9)
#     parser.add_argument('--epsilon', type=float, help='Initial epsilon for exploration', default=0.99)
#     parser.add_argument('--epsilon_min', type=float, help='Minimum epsilon for exploration', default=0.01)
#     parser.add_argument('--epsilon_decay', type=float, help='Epsilon decay rate', default=0.9995)
#     parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
#     parser.add_argument('--batch_size', type=int, help='Batch size for replay', default=64)
#     parser.add_argument('--memory_size', type=int, help='Replay memory size', default=10000)
#     parser.add_argument('--noise_std', type=float, help='Standard deviation of noise', default=0.1)
#     args = parser.parse_args()

#     hyperparams = HyperParameters(
#         update_rate=args.update_rate,
#         gamma=args.gamma,
#         epsilon=args.epsilon,
#         epsilon_min=args.epsilon_min,
#         epsilon_decay=args.epsilon_decay,
#         lr=args.lr,
#         batch_size=args.batch_size,
#         memory_size=args.memory_size,
#         reward_fun=args.reward,
#         episodes=args.episodes,
#         noise_std=args.noise_std,
#         test_episodes=args.test_episodes
#     )

#     agent = DQNAgent(hyperparams)

#     if args.load:
#         agent.load_model(args.load)

#     if args.test:
#         agent.test(model_filename=args.load, episodes=hyperparams.test_episodes)
#     else:
#         agent.train(postfix=args.save, episodes=hyperparams.episodes)
