import matplotlib.pyplot as plt
import numpy as np

# Parameters
initial_epsilon = 0.99
num_episodes = 100000
decay_factors = [0.999, 0.9999, 0.99993]

# Function to calculate epsilon decay
def calculate_epsilon_decay(initial_epsilon, decay_factor, num_episodes):
    epsilon_values = []
    epsilon = initial_epsilon
    for episode in range(num_episodes):
        epsilon_values.append(epsilon)
        epsilon *= decay_factor
    return epsilon_values

# Plotting
plt.figure(figsize=(14, 8))

for decay_factor in decay_factors:
    epsilon_values = calculate_epsilon_decay(initial_epsilon, decay_factor, num_episodes)
    plt.plot(epsilon_values, label=f'Decay Factor: {decay_factor}')
    for i in range(0, num_episodes, 1000):
        if epsilon_values[i] >= 0.01:
            plt.annotate(f'{epsilon_values[i]:.2f}', (i, epsilon_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
        print(f'Decay Factor {decay_factor} - Episode {i}: Epsilon = {epsilon_values[i]:.5f}')

plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay over Episodes for Different Decay Factors')
plt.legend()
plt.grid(True)
plt.savefig('epsilon_decay_plot.png')
plt.show()
