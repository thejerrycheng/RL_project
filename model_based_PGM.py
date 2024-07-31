import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import math

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
        self.fc5 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        action_probs = torch.softmax(self.fc5(x), dim=-1)
        return action_probs

# Define the model-based agent
class ModelBasedAgent:
    def __init__(self, env, dynamics_lr=0.001, policy_lr=0.0001, gamma=0.99):
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
        self.memory = deque(maxlen=10000)
        self.loss = math.inf

        # Create directory for saving models
        self.model_save_path = "saved_policy_models"
        os.makedirs(self.model_save_path, exist_ok=True)

         # Create directory for saving models
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
        # print("State:", state)
        # Ensure the state is a 2D tensor (batch size of 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Forward pass through the policy network to get action probabilities
        action_probs = self.policy_network(state).squeeze(0)  # Remove the batch dimension
        
        # Check shape of action_probs for debugging
        # print("Action probabilities:", action_probs.numpy())
        
        # Convert action probabilities to a 1D NumPy array
        action_probs = action_probs.cpu().detach().numpy()
        
        # Ensure probabilities sum to 1 by adding a small value and normalizing
        action_probs = action_probs / np.sum(action_probs)
        
        # Print action_probs to ensure they're correctly shaped
        # print("Normalized action probabilities:", action_probs)
        
        # Choose an action based on the probabilities
        action = np.random.choice(self.action_dim, p=action_probs)
        
        return action


    def train_policy(self, num_episodes=100000):
        rewards_per_episode = []  # Store total rewards for each episode
        highest_reward = -float('inf')  # Initialize the highest reward

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0  # Total reward for the current episode
            while not done:
                action = self.select_action(state)

                next_state, reward, done, _, _ = self.env.step(action)

                # Simulate with the dynamics model
                # predicted_next_state, predicted_reward = self.dynamics_model(
                #     torch.FloatTensor(state).to(self.device),  # Ensure state is 2D
                #     torch.LongTensor(action).to(self.device)  # Ensure action is 2D
                # )

                # Accumulate rewards
                # total_reward += predicted_reward
                total_reward += reward

                # state = predicted_next_state.detach().squeeze(0).numpy()  # Ensure state is a NumPy array
                state = next_state

            # Save the total reward for this episode
            rewards_per_episode.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Total loss: {self.loss}")

            # Save the model if a new highest reward is found
            if total_reward > highest_reward:
                highest_reward = total_reward
                model_filename = f"policy_model_highest_reward_{highest_reward:.4f}.pth"
                model_filepath = os.path.join(self.model_save_path, model_filename)
                torch.save(self.policy_network.state_dict(), model_filepath)
                print(f"New highest reward {highest_reward:.4f} found, model saved to {model_filepath} ------------------------")

            # Update policy every 10 episodes
            # if (episode + 1) % 2 == 0:
            self.update_policy(rewards_per_episode[-1:])  # Pass the last 2 episodes' rewards

        # Save rewards for plotting
        self.rewards_per_episode = rewards_per_episode

    def update_policy(self, recent_rewards):
        # Calculate the mean reward over the recent episodes
        mean_reward = np.mean(recent_rewards)

        # Sample states from memory to update the policy
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, 64))
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        # Forward pass through the policy network
        action_probs = self.policy_network(states)

        # Calculate the log probabilities of the taken actions
        log_probs = torch.log(action_probs[range(len(actions)), actions])

        # Update the policy network using the mean reward as a baseline
        loss = -(log_probs * mean_reward).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        # print(f"Updated policy with mean reward: {mean_reward}")



    def reward_function(self, state):
        # Define your custom reward function here
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)
        return reward

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Dynamics Model Training Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.rewards_per_episode)), self.rewards_per_episode, label='Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = ModelBasedAgent(env)

    # Collect initial data
    agent.collect_data()

    # Train dynamics model
    agent.train_dynamics_model()

    # Plot the training loss
    # agent.plot_loss()

    # state, _ = env.reset()
    # print("Initial state:", state)
    # done = False
    # total_reward = 0  # Total reward for the current episode
    # action = agent.select_action(state)
    # print("Selected action:", action)
    # Simulate with the dynamics model
    # predicted_next_state, predicted_reward = agent.dynamics_model(
    #     torch.FloatTensor(state).unsqueeze(0).to(agent.device),  # Ensure state is 2D
    #     torch.LongTensor([action]).unsqueeze(0).to(agent.device)  # Ensure action is 2D
    # )
    # print("Predicted next state:", predicted_next_state)
    # print("Predicted reward:", predicted_reward)
    # print("The state input tensor looks like", torch.FloatTensor(state).to(agent.device)) # this is the correct dimension
    # print("The action input tensor looks like", torch.LongTensor(action).unsqueeze(1).to(agent.device)) # this is the correct dimension
    # print("The state tensor is on the device:", predicted_next_state.detach().squeeze(0).numpy()) # this is the correct dimension
    # print("The reward tensor is on the device:", predicted_reward.detach().numpy()) # this is the correct dimension

    
    # Train policy
    agent.train_policy()

    # Plot the rewards
    agent.plot_rewards()

    # Test policy
    # for _ in range(10):
    #     state, _ = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action = agent.select_action(state)
    #         state, reward, done, _, _ = env.step(action)
    #         total_reward += reward
    #     print(f"Total reward: {total_reward}")
