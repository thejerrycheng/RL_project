import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_rolling_average_with_bounds(groups, window=100):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (method, file_paths) in enumerate(groups.items()):
        all_rewards = []
        
        # Read and store rolling average rewards from each file
        for file_path in file_paths:
            data = pd.read_csv(file_path)
            rolling_avg = data['Total Reward'].rolling(window=window).mean()
            all_rewards.append(rolling_avg.values)
        
        # Convert list of rewards to a DataFrame
        rewards_df = pd.DataFrame(all_rewards).T
        
        # Calculate mean, upper bound, and lower bound
        mean_rewards = rewards_df.mean(axis=1)
        upper_bounds = rewards_df.max(axis=1)
        lower_bounds = rewards_df.min(axis=1)
        
        # Plotting
        episodes = np.arange(len(mean_rewards))
        color = colors[i % len(colors)]
        
        plt.plot(episodes, mean_rewards, label=f'Average Reward ({method})', color=color)
        plt.fill_between(episodes, lower_bounds, upper_bounds, color=color, alpha=0.3, label=f'Bounds ({method})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Rolling Average Reward (window={window}) with Bounds')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rolling average rewards with bounds from multiple CSV files grouped by methods.")
    parser.add_argument('--groups', metavar='G', type=str, nargs='+', help='CSV file paths grouped by method in the format method:file1.csv,file2.csv,...', required=True)
    parser.add_argument('--window', type=int, default=100, help='Rolling window size for averaging')
    args = parser.parse_args()

    # Parse the groups argument into a dictionary
    groups = {}
    for group in args.groups:
        method, files = group.split(':')
        file_paths = files.split(',')
        groups[method] = file_paths

    plot_rolling_average_with_bounds(groups, args.window)



# python plot_grouped_bounds.py --groups DQN:file1.csv,file2.csv,file3.csv PPO:file4.csv,file5.csv,file6.csv --window 100
