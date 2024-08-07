import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_average_reward_over_episodes(file_paths, window):
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        data['Average Reward'] = data['Total Reward'].rolling(window=window).mean()
        plt.plot(data['Episode'], data['Average Reward'], label=file_path)
    
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (window={window})')
    plt.title(f'Average Reward over {window} Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average rewards over episodes from multiple CSV files.")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='CSV file paths')
    parser.add_argument('--window', type=int, default=1000, help='Window length for rolling average')
    args = parser.parse_args()
    
    plot_average_reward_over_episodes(args.files, args.window)
