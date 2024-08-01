import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import math


class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2,  # cart position range is [-2.4, 2.4]
            0.5,  # cart velocity range is approximated as [-50, 50]
            math.radians(20),  # pole angle range is [-12 degrees, 12 degrees]
            math.radians(0.5)  # pole angular velocity range is approximated as [-50 degrees/s, 50 degrees/s]
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs


# Define the dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)  # Output next state
        self.fc4 = nn.Linear(256, 1)  # Output reward

    def forward(self, state, action):
        # Ensure state is a 2D tensor
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Ensure action is a 2D tensor
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # One-hot encode the action
        action_one_hot = torch.zeros((action.size(0), self.fc1.in_features - state.size(1)), device=action.device)
        action_one_hot.scatter_(1, action, 1)

        # Concatenate state and one-hot encoded action
        x = torch.cat([state, action_one_hot], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output next state and reward
        next_state = self.fc3(x) # Ensure next_state is a 2D tensor
        reward = self.fc4(x).squeeze(-1)  # Ensure reward is a 1D tensor

        return next_state, reward

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

# Define the model-based agent
class ModelBasedAgent:
    def __init__(self, env, dynamics_lr=0.001, policy_lr=0.00001, gamma=0.99, clip_param=0.2, ppo_epochs=10, batch_size=64):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dynamics_model = DynamicsModel(self.state_dim, self.action_dim).to(self.device)
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=dynamics_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.gamma = gamma
        self.loss = 0
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)

        # Create directory for saving models
        self.model_save_path = "saved_policy_models"
        os.makedirs(self.model_save_path, exist_ok=True)

        # Create directory for saving dynamics models
        self.dynamics_model_save_path = "saved_dynamics_models"
        os.makedirs(self.dynamics_model_save_path, exist_ok=True)

        # Initialize lists to store losses and epochs
        self.losses = []
        self.epochs = []
        self.rewards_per_episode = []  # List to store rewards for each episode

    def collect_data(self, num_episodes=100):
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

    def train_dynamics_model(self, batch_size=64, epochs=1000, save_interval=100):
        min_loss = float('inf')  # Initialize minimum loss

        for epoch in range(epochs):
            minibatch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, _ = zip(*minibatch)

            # Convert lists of numpy arrays to a single numpy array
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)

            # Convert numpy arrays to PyTorch tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # Ensure actions are 2D
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

            predicted_next_states, predicted_rewards = self.dynamics_model(states, actions)

            # Compute losses for both next state and reward
            state_loss = nn.MSELoss()(predicted_next_states, next_states)
            reward_loss = nn.MSELoss()(predicted_rewards, rewards)
            total_loss = state_loss + reward_loss

            self.dynamics_optimizer.zero_grad()
            total_loss.backward()
            self.dynamics_optimizer.step()

            # Save loss and epoch
            self.losses.append(total_loss.item())
            self.epochs.append(epoch + 1)

            # Print out the log
            print(f"Epoch: {epoch + 1}, State Loss: {state_loss.item()}, Reward Loss: {reward_loss.item()}, Total Loss: {total_loss.item()}")

            # Save the model at specified intervals or when a new minimum loss is achieved
            if (epoch + 1) % save_interval == 0 or total_loss.item() < min_loss:
                min_loss = total_loss.item()
                model_filename = f"dynamics_model.pth"
                model_filepath = os.path.join(self.dynamics_model_save_path, model_filename)
                torch.save(self.dynamics_model.state_dict(), model_filepath)
                print(f"Model saved to {model_filepath}")

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.policy_network(state)
        action_probs = action_probs.squeeze(0).cpu().detach().numpy()
        action_probs = action_probs / np.sum(action_probs)
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def train_policy(self, num_episodes=5000):
        rewards_per_episode = []  # Store total rewards for each episode
        highest_reward = -float('inf')  # Initialize the highest reward

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0  # Total reward for the current episode
            states = []
            actions = []
            rewards = []
            state_values = []
            log_probs = []
            step = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # Custom reward function - reward reshaping 
                reward = self.reward_function(next_state)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, state_value = self.policy_network(state_tensor)
                state_values.append(state_value.item())
                log_prob = torch.log(action_probs.squeeze(0)[action])
                log_probs.append(log_prob.item())

                total_reward += reward
                state = next_state
                step += 1

                if step > 1000:
                    print("Success")
                    break

            rewards_per_episode.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            if total_reward > highest_reward:
                highest_reward = total_reward
                model_filename = f"ppo_highest_2000.pth"
                model_filepath = os.path.join(self.model_save_path, model_filename)
                torch.save(self.policy_network.state_dict(), model_filepath)
                print(f"New highest reward {highest_reward:.4f} found, model saved to {model_filepath}, Loss: {self.loss}")

            self.update_policy(states, actions, rewards, state_values, log_probs)

        self.rewards_per_episode = rewards_per_episode
        
        # Save the final model
        final_model_filename = "ppo_final_2000.pth"
        final_model_filepath = os.path.join(self.model_save_path, final_model_filename)
        torch.save(self.policy_network.state_dict(), final_model_filepath)
        print(f"Final model saved to {final_model_filepath}")

    def update_policy(self, states, actions, rewards, state_values, log_probs):
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        state_values = torch.FloatTensor(state_values).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)

        advantages = discounted_rewards - state_values
        old_log_probs = log_probs.detach()

        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                sampled_indices = slice(i, i + self.batch_size)
                sampled_states = torch.FloatTensor(states[sampled_indices]).to(self.device)
                sampled_actions = torch.LongTensor(actions[sampled_indices]).to(self.device)
                sampled_advantages = advantages[sampled_indices].detach()
                sampled_old_log_probs = old_log_probs[sampled_indices]

                action_probs, state_values = self.policy_network(sampled_states)
                log_probs = torch.log(action_probs[range(len(sampled_actions)), sampled_actions])
                ratio = torch.exp(log_probs - sampled_old_log_probs)
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * sampled_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(state_values.squeeze(-1), discounted_rewards[sampled_indices])

                self.policy_optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.policy_optimizer.step()

    def reward_function(self, state):
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) 
        return reward

    def plot_loss(self, save_path='dynamics_training_loss.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Dynamics Model Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Training loss plot saved to {save_path}")

    def plot_rewards(self, save_path='ppo_rewards_per_episode_2000.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.rewards_per_episode)), self.rewards_per_episode, label='Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Rewards per episode plot saved to {save_path}")

    def load_model(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy_network.eval()
        print(f"Loaded model from {filepath}")
    
    def test_policy(agent, env_name='CartPole-v1', testing_episodes=10):
        # Create environment for testing
        env = gym.make(env_name, render_mode=None)
        env = NoisyObservationWrapper(env, noise_std=0.1)
        test_rewards = []  # List to store rewards for each test episode
        success_num = 0

        for episode in range(testing_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action = agent.select_action(state)
                state, reward, done, _, _ = env.step(action)

                reward = agent.reward_function(state)

                total_reward += reward
                step += 1
                if step > 1000:
                    print("Success")
                    success_num += 1
                    break
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

        print("The success rate is: ", (success_num / testing_episodes) * 100, "%")    
        
        # Plot the test rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(test_rewards) + 1), test_rewards, marker='o', label='Test Rewards')
        plt.xlabel('Test Episode')
        plt.ylabel('Total Reward')
        plt.title('Test Rewards per Episode')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model-Based RL with PPO')
    parser.add_argument('--load', type=str, help='Path to the pre-trained policy model')
    args = parser.parse_args()

    env = gym.make('CartPole-v1')
    env = NoisyObservationWrapper(env, noise_std=0.2)
    agent = ModelBasedAgent(env)

    if args.load:
        agent.load_model(args.load)

        agent.test_policy()
    else:
        # Collect initial data
        agent.collect_data()

        # Train dynamics model
        agent.train_dynamics_model()

        # Plot the training loss
        agent.plot_loss()

        # Train policy
        agent.train_policy()

        # Plot the rewards
        agent.plot_rewards()

        # Test policy
        agent.test_policy()
 
