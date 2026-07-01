import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_full_curriculum(jsonl_paths, stage_transitions):
    # 1. Load all data files and combine them
    all_data = []
    for path in jsonl_paths:
        with open(path, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    df = pd.DataFrame(all_data)
    # Sort by global_step to ensure the line isn't tangled
    df = df.sort_values('global_step')
    
    # 2. Setup the plot
    plt.figure(figsize=(12, 6))
    
    # 3. Calculate rolling mean (window=50 captures the trends better for 200k+ steps)
    df['smoothed_reward'] = df['total_reward'].rolling(window=50).mean()
    
    # 4. Plot the curve
    plt.plot(df['global_step'], df['smoothed_reward'], label='Smoothed Reward', color='blue', linewidth=2)
    
    # 5. Add vertical markers for stages
    for step in stage_transitions:
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.6)
        
    plt.text(stage_transitions[0], df['smoothed_reward'].max(), ' Stage 1 (Sparse)', rotation=90, color='red')
    plt.text(stage_transitions[1], df['smoothed_reward'].max(), ' Stage 2 (Mid)', rotation=90, color='red')
    plt.text(stage_transitions[2], df['smoothed_reward'].max(), ' Stage 3 (Dense)', rotation=90, color='red')
    
    plt.title('Full Curriculum Learning Convergence')
    plt.xlabel('Global Training Steps')
    plt.ylabel('Total Episodic Reward')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Update these to match the actual step counts where your config changed stages
# Based on your previous logs, your stages were roughly at 50k, 100k, and 150k
stage_steps = [50000, 100000, 150000]

# List all your episode files here
plot_full_curriculum(['logs/episodes.jsonl'], stage_steps)