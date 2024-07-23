# Agent.py

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
    

######### YOU DON'T NEED TO MODIFY THIS CLASS -- IT'S ALREADY COMPLETED ###############
class Replay_Buffer():
    """
    Experience Replay Buffer to store experiences
    """
    def __init__(self, size, device):
        self.device = device
        self.size = size # size of the buffer
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)
        
    def store(self, state, action, next_state, reward, terminal):
        """
        Store experiences to their respective queues
        """      
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        
    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)
        return states, actions, next_states, rewards, terminals
    
    def __len__(self):
        return len(self.terminals)
    
    
class Agent:
    """
    Implementing Agent DQL Algorithm
    """
    def __init__(self, env: gym.Env, hyperparameters: Hyperparameters, device=False):
        # Some Initializations
        if not device:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters  
        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0
        self.action_space = env.action_space
        self.feature_space = env.observation_space
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device=self.device)
        
        # Initiate the online and Target DQNs
        self.onlineDQN = DQN(self.action_space.n, self.feature_space.n).to(self.device)
        self.targetDQN = DQN(self.action_space.n, self.feature_space.n).to(self.device)
        self.update_target()  # Ensure the target network starts with the same weights as the online network

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)
                
    def epsilon_greedy(self, state):
        """
        Implement epsilon-greedy policy
        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.onlineDQN(state)
            return torch.argmax(q_values).item()

    def greedy(self, state):
        """
        Implement greedy policy
        """ 
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.onlineDQN(state)
        return torch.argmax(q_values).item()
   
    def apply_SGD(self, ended):
        """
        Train DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """ 
        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)
                    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)       

        Q_hat = self.onlineDQN(states).gather(1, actions)

        with torch.no_grad():   
            next_target_q_value = self.targetDQN(next_states).max(1, keepdim=True)[0]
        
        next_target_q_value[terminals] = 0
        y = rewards + (self.hp.discount_factor * next_target_q_value)

        loss = self.loss_function(Q_hat, y) # Compute the loss
        
        # Update the running loss and learned counts for logging and plotting
        self.current_loss += loss.item()
        self.episode_counts += 1

        if ended:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss) 
            self.current_loss = 0
            self.episode_counts = 0
        
        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        
        self.optimizer.step()
 


    ############## THE REMAINING METHODS HAVE BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def update_target(self):
        """
        Update the target network 
        """
        # Copy the online DQN into target DQN
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

    def update_epsilon(self):
        """
        Reduce epsilon by the decay factor
        """
        # Gradually reduce epsilon
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)
        
    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extension
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)



# Question 3 - Implement the DQL class, on the feature_representation method, train() and play() methods
class DQL():
    def __init__(self, hyperparameters: Hyperparameters, train_mode):

        if train_mode:
            render = None
        else:
            render = "human"

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters

        # Load the environment
        self.env = gym.make('FrozenLake-v1', map_name=f"{self.hp.map_size}x{self.hp.map_size}", is_slippery=False, render_mode=render)

        # Initiate the Agent
        self.agent = Agent(env=self.env, hyperparameters=self.hp)
                
    def feature_representation(self, state: int):
        """
        Represent the feature of the state
        We simply use tokenization
        """
        feature = np.zeros(self.env.observation_space.n, dtype=np.float32)
        # print("The state is: ", state)
        feature[state] = 1
        return feature
    

    # Exercise for fun: Implement a custom reward shaping function
    def shaped_reward(self, state, reward, done):
        """
        Define a custom reward function for reward shaping
        """
        goal_position = self.env.observation_space.n - 1  # Assuming the goal is the last state
        hole_positions = [5, 7, 11, 12]  # Example positions of holes in a 4x4 grid
        
        if state == goal_position:
            return 10.0  # Large positive reward for reaching the goal
        elif state in hole_positions:
            return -10.0  # Large negative reward for falling into a hole
        elif done:
            return -1.0  # Penalty for ending the episode without reaching the goal
        else:
            return reward - 0.1  # Small penalty for each step to encourage shorter paths

    
    def train(self): 
        """                
        Train the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in range(1, self.hp.num_episodes + 1):
            
            # sample a new state
            state, _ = self.env.reset()
            state = self.feature_representation(int(state))
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0
                                                
            while not ended and not truncated:
                # Find action via epsilon greedy 
                # use what you implemented in Class Agent
                action = self.agent.epsilon_greedy(state)

                # Find next state and reward
                next_state, reward, ended, truncated, _ = self.env.step(action)

                # Apply shaped reward if needed
                if self.shaped_reward == True:
                    reward = self.shaped_reward(next_state, reward, ended)

                # Find the feature of next_state using your implementation self.feature_representation
                next_state = self.feature_representation(int(next_state))
                
                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, ended) 
                
                if len(self.agent.replay_buffer) > self.hp.batch_size and sum(self.collected_rewards) > 0:
                    # Use self.agent.apply_SGD implementation to update the online DQN
                    self.agent.apply_SGD(ended)
                    
                    # Update target-network weights
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        # Copy the online DQN into the Target DQN using what you implemented in Class Agent
                        self.agent.update_target()
                
                state = next_state
                episode_reward += reward
                step_size += 1
                            
            self.collected_rewards.append(episode_reward)                     
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                      f"Total Time Steps: {total_steps}, "
                      f"Trajectory Length: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")
            print(printout)
        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curves()
                                                                    

    def play(self):  
        """                
        Play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        print("The model is loaded from ", self.hp.RL_load_path)
        self.agent.onlineDQN.eval()
        
        # Playing 
        for episode in range(1, self.hp.num_test_episodes + 1):         
            state, _ = self.env.reset()
            state_feature = self.feature_representation(int(state))
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0
                                                           
            while not ended and not truncated:
                # Find the feature of state using your implementation self.feature_representation
                # Act greedy and find action using what you implemented in Class Agent
                action = self.agent.greedy(state_feature)
                
                next_state, reward, ended, truncated, _ = self.env.step(action)

                # Apply shaped reward if needed
                if self.shaped_reward == True:
                    reward = self.shaped_reward(next_state, reward, ended)
                    
                next_state = self.feature_representation(int(next_state))
                
                state_feature = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
            
        pygame.quit()
        
    ############## THIS METHOD HAS BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(100)/100, mode='valid')
        
        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./{self.hp.map_size}x{self.hp.map_size}/low_epislon_Reward_vs_Episode_{self.hp.map_size}_x_{self.hp.map_size}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close() 
        
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")
        
        # Save the figure
        plt.savefig(f'./{self.hp.map_size}x{self.hp.map_size}/low_epislon_Learning_Curve_{self.hp.map_size}_x_{self.hp.map_size}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


