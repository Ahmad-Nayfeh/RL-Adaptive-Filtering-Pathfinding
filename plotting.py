# plotting.py (v8 - composite generator, adapted for .npy files)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import glob # Added for glob
import csv # For potential future use if CSV generation is also added here.

try:
    import config
    PLOTS_DIR = getattr(config, 'PLOTS_DIR', 'plots')
    RESULTS_DIR = getattr(config, 'RESULTS_DIR', 'results') # Added for results path
    CSV_SUMMARY_FILENAME = getattr(config, 'CSV_SUMMARY_FILENAME', 'experiment_summary.csv')
except ImportError:
    PLOTS_DIR = 'plots'
    RESULTS_DIR = 'results'
    CSV_SUMMARY_FILENAME = 'experiment_summary.csv'

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results dir exists for summary CSV

ALG_COLORS = {
    'LMS': '#1f77b4',
    'NLMS': '#2ca02c',
    'SIGN_ERROR_LMS': '#ff7f0e',
    'RLS': '#d62728',        # Generic RLS
    'RLS_LSTD': '#9467bd',   # Specific RLS_LSTD
    'DEFAULT': '#333333'
}

def _moving_average(x, win=50):
    if not isinstance(x, np.ndarray): x = np.array(x)
    if x.size == 0: return np.array([]) # Handle empty array
    if win <= 1 or len(x) < win:
        return x
    return np.convolve(x, np.ones(win)/win, mode='valid')

def _draw_grid(ax, rows, cols, obstacles, start_pos, goal_pos):
    ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(-0.5, rows-0.5); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    cmap_cells = {'empty':'white', 'obstacle':'#404040', 'start':'green', 'goal':'red'}
    for r in range(rows):
        for c in range(cols):
            color = (
                cmap_cells['obstacle'] if (r,c) in obstacles else
                (cmap_cells['start'] if (r,c)==start_pos else
                 (cmap_cells['goal'] if (r,c)==goal_pos else cmap_cells['empty']))
            )
            ax.add_patch(mpatches.Rectangle((c-0.5, r-0.5), 1, 1, # x, y are col, row
                                            facecolor=color, edgecolor='#BBBBBB', linewidth=0.5))
    ax.invert_yaxis() # so (0,0) at top-left if plotting (row,col), but imshow origin='lower' is often used

def _draw_path(ax, path, color, lw=1.5, label=None, marker='.', markersize=3):
    if not path or len(path) < 1: return
    # Path is list of (row, col) tuples. For plotting, x is col, y is row.
    cols_path = [p[1] for p in path]
    rows_path = [p[0] for p in path]
    ax.plot(cols_path, rows_path, color=color, linewidth=lw, marker=marker, markersize=markersize, label=label)

def _plot_value_heatmap(ax, v_grid, title='State-Value Heatmap V(s)'):
    if v_grid is None:
        ax.axis('off'); ax.text(0.5,0.5,'No value grid data', ha='center', va='center')
        return
    im = ax.imshow(v_grid, origin='lower', cmap='viridis', aspect='auto') # origin='lower' common for heatmaps
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def save_composite_figure(result_dict, palette=ALG_COLORS):
    algo = result_dict.get('algorithm','UnkAlgo')
    grid_s = result_dict.get('grid_size','?x?')
    feat = result_dict.get('feature_strategy','UnkFeat')
    eps_run = result_dict.get('num_episodes_configured','UnkEps')
    
    rewards = result_dict.get('episode_rewards', [])
    v_grid = result_dict.get('value_function_grid', None)
    learned_path = result_dict.get('learned_path', [])
    optimal_path = result_dict.get('optimal_path_bfs', []) # Changed key to match main_experiment.py
    
    # Get obstacles, start_pos, goal_pos from result_dict (snapshot of config)
    # This is more robust than relying on global config at plotting time.
    config_snapshot = result_dict.get('config_snapshot', {})
    
    if 'x' in grid_s:
        rows, cols = map(int, grid_s.split('x'))
    else: # Fallback if grid_size format is unexpected
        rows = config_snapshot.get('GRID_ROWS', 10) # Default to some value
        cols = config_snapshot.get('GRID_COLS', 10)

    # Determine actual start/goal/obstacles used for THIS run from its config_snapshot
    if rows == 16 and cols == 16: # Logic from your GridEnvironment
        start_pos = tuple(config_snapshot.get('START_POS', (0,0)))
        goal_pos = tuple(config_snapshot.get('GOAL_POS', (rows-1, cols-1)))
        current_obstacle_list = config_snapshot.get('OBSTACLES_16x16', [])
    elif rows == 8 and cols == 8:
        start_pos = tuple(config_snapshot.get('START_POS_8x8', (0,0)))
        goal_pos = tuple(config_snapshot.get('GOAL_POS_8x8', (rows-1, cols-1)))
        current_obstacle_list = config_snapshot.get('OBSTACLES_8x8', [])
    else: # Fallback
        start_pos = (0,0)
        goal_pos = (rows-1, cols-1)
        current_obstacle_list = []
    
    obstacles = set()
    for r_obs, c_obs in current_obstacle_list:
        if 0 <= r_obs < rows and 0 <= c_obs < cols:
            if (r_obs, c_obs) != start_pos and (r_obs, c_obs) != goal_pos:
                obstacles.add((r_obs, c_obs))

    color = palette.get(algo.upper(), palette['DEFAULT'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10)) # Slightly larger figure
    fig.suptitle(f"Summary: {algo} on {grid_s} ({feat}) - {eps_run} Episodes", fontsize=14, weight='bold')

    # (0,0) Learning curve
    if rewards:
        win = max(1, int(0.02 * len(rewards))) # Moving average window 2% of episodes
        smoothed_rewards = _moving_average(rewards, win)
        # Adjust x-axis for smoothed rewards to align properly
        x_smooth = np.arange(win - 1, len(rewards)) if len(rewards) >= win else np.arange(len(rewards))
        x_raw = np.arange(len(rewards))
        
        axs[0,0].plot(x_raw, rewards, color=color, alpha=0.3, label=f'Raw (Episode Reward)')
        if smoothed_rewards.size > 0 :
             axs[0,0].plot(x_smooth, smoothed_rewards, color=color, linewidth=2, label=f'Smoothed (Window {win})')
        axs[0,0].set_title('Learning Curve'); axs[0,0].set_xlabel('Episode'); axs[0,0].set_ylabel('Cumulative Reward')
        axs[0,0].legend(fontsize='small'); axs[0,0].grid(True, alpha=0.5)
    else:
        axs[0,0].axis('off'); axs[0,0].text(0.5,0.5,'No reward data', ha='center', va='center')

    # (0,1) Value heatmap
    _plot_value_heatmap(axs[0,1], v_grid, title=f'State-Value Heatmap $V(s)$')
    if v_grid is not None: axs[0,1].invert_yaxis() # Match grid orientation if imshow is used

    # (1,0) Grid map with paths
    if rows and cols:
        _draw_grid(axs[1,0], rows, cols, obstacles, start_pos, goal_pos)
        # Plot paths (ensure correct color from config or palette)
        optimal_path_color = config.COLOR_OPTIMAL_PATH if hasattr(config, 'COLOR_OPTIMAL_PATH') else '#00FFFF' # Cyan
        learned_path_color = color # Use algorithm's color
        
        _draw_path(axs[1,0], optimal_path, color=optimal_path_color, lw=2, label=f'Optimal ({len(optimal_path)-1 if optimal_path else 0} steps)', marker='x', markersize=5)
        _draw_path(axs[1,0], learned_path, color=learned_path_color, lw=1.5, label=f'Learned ({len(learned_path)-1 if learned_path else 0} steps)', marker='o', markersize=3)
        axs[1,0].legend(fontsize='small', loc='best') # 'best' or specific location
        axs[1,0].set_title(f'Paths on {grid_s} Grid')
    else:
        axs[1,0].axis('off'); axs[1,0].text(0.5,0.5,'No grid info', ha='center', va='center')

    # (1,1) Path length ratio bar & Other key stats
    path_ratio = result_dict.get('path_length_ratio', None)
    final_avg_reward = result_dict.get('avg_reward_final_episodes', None)
    training_time = result_dict.get('training_duration_seconds', None)
    
    stats_text = []
    if path_ratio is not None:
        stats_text.append(f"Path Ratio (L/O): {path_ratio:.2f}")
    if final_avg_reward is not None:
        stats_text.append(f"Avg Final Reward: {final_avg_reward:.2f}")
    if training_time is not None:
        stats_text.append(f"Training Time: {training_time:.2f} s")
    
    if stats_text:
        axs[1,1].text(0.05, 0.95, "\n".join(stats_text), transform=axs[1,1].transAxes,
                      fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        axs[1,1].axis('off')
        axs[1,1].set_title('Key Metrics Summary', fontsize=10)
    else:
        axs[1,1].axis('off'); axs[1,1].text(0.5,0.5,'No summary metrics', ha='center', va='center')


    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    # Sanitize feature strategy for filename (e.g., replace underscores or shorten)
    feat_fname = feat.replace('_','-')
    fname = f"SUMMARY_{algo}_{feat_fname}_{grid_s}_{eps_run}eps.png"
    out_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    if config.VERBOSE_LOGGING: print(f"Saved composite figure: {out_path}")
    return out_path

def generate_csv_summary(results_data_list):
    if not results_data_list:
        if config.VERBOSE_LOGGING: print("No data provided for CSV summary.")
        return
    
    # Define fieldnames based on keys in your results_data dictionary from main_experiment.py
    # Ensure these match what's actually saved.
    fieldnames = [
        'Algorithm', 'Feature_Strategy', 'Grid_Size', 
        'Num_Episodes_Configured', 'Max_Steps_Per_Episode_Configured',
        'Total_Training_Steps_Executed', 'Training_Duration_Seconds',
        'Avg_Reward_Overall', 'Avg_Steps_Overall', 
        'Avg_Reward_Final_Episodes', 'Avg_Steps_Final_Episodes',
        'Learned_Path_Length', 'Optimal_Path_Length', 'Path_Length_Ratio',
        # Add parameter fields from config_snapshot if you want them in CSV
        'LMS_LR', 'NLMS_LR', 'NLMS_Epsilon_Norm', 'SignError_LR', 
        'RLS_Lambda', 'RLS_P_Init', 'Gamma', 'Epsilon_Decay_Steps', 'Random_Seed'
    ]
    
    csv_path = os.path.join(RESULTS_DIR, CSV_SUMMARY_FILENAME)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for data in results_data_list:
            row = {
                'Algorithm': data.get('algorithm'),
                'Feature_Strategy': data.get('feature_strategy'),
                'Grid_Size': data.get('grid_size'),
                'Num_Episodes_Configured': data.get('num_episodes_configured'),
                'Max_Steps_Per_Episode_Configured': data.get('max_steps_per_episode_configured'),
                'Total_Training_Steps_Executed': data.get('total_training_steps_executed'),
                'Training_Duration_Seconds': f"{data.get('training_duration_seconds', 0):.2f}",
                'Avg_Reward_Overall': f"{data.get('avg_reward_overall', 0):.2f}",
                'Avg_Steps_Overall': f"{data.get('avg_steps_overall', 0):.2f}",
                'Avg_Reward_Final_Episodes': f"{data.get('avg_reward_final_episodes', 0):.2f}",
                'Avg_Steps_Final_Episodes': f"{data.get('avg_steps_final_episodes', 0):.2f}",
                'Learned_Path_Length': data.get('learned_path_length', -1),
                'Optimal_Path_Length': data.get('optimal_path_length', -1), # Changed from optimal_path_bfs length
                'Path_Length_Ratio': f"{data.get('path_length_ratio', -1):.2f}",
            }
            # Add config snapshot parameters
            cfg_snap = data.get('config_snapshot', {})
            row.update({
                'LMS_LR': cfg_snap.get('LMS_LEARNING_RATE'),
                'NLMS_LR': cfg_snap.get('NLMS_LEARNING_RATE'),
                'NLMS_Epsilon_Norm': cfg_snap.get('NLMS_EPSILON_NORM'),
                'SignError_LR': cfg_snap.get('SIGN_ERROR_LMS_LEARNING_RATE'),
                'RLS_Lambda': cfg_snap.get('RLS_FORGETTING_FACTOR'),
                'RLS_P_Init': cfg_snap.get('RLS_INIT_P_DIAG_VALUE'),
                'Gamma': cfg_snap.get('GAMMA'),
                'Epsilon_Decay_Steps': cfg_snap.get('EPSILON_DECAY_STEPS'),
                'Random_Seed': cfg_snap.get('RANDOM_SEED')
            })
            writer.writerow(row)
    if config.VERBOSE_LOGGING: print(f"CSV summary saved to {csv_path}")


def generate_all_plots_and_summary(results_dir_override=None):
    """Walk the results directory, load *.npy results files, and spit out composites."""
    # Use override if provided, else use from config or default
    current_results_dir = results_dir_override if results_dir_override else RESULTS_DIR

    # MODIFIED TO LOOK FOR .npy FILES
    files = glob.glob(os.path.join(current_results_dir, '**', '*.npy'), recursive=True)
    if not files:
        if config.VERBOSE_LOGGING: print(f"No .npy result files found under {current_results_dir}")
        return
    
    if config.VERBOSE_LOGGING: print(f"Found {len(files)} .npy result files in {current_results_dir} -> generating composite figures and CSV summary...")
    
    all_results_data = []
    for fpath in files:
        try:
            # Assumes .npy file stores a single dictionary item
            data = np.load(fpath, allow_pickle=True).item() 
            all_results_data.append(data) # Collect data for summary CSV
            out_path = save_composite_figure(data) # Generate composite plot for this run
            if config.VERBOSE_LOGGING: print(f"✔ Processed {os.path.basename(fpath)} → {os.path.basename(out_path)}")
        except Exception as e:
            print(f"✘ Error processing file {fpath}: {e}")
            if config.VERBOSE_LOGGING: import traceback; traceback.print_exc()

    if all_results_data:
        generate_csv_summary(all_results_data) # Generate one CSV from all loaded results
    else:
        if config.VERBOSE_LOGGING: print("No data successfully loaded to generate CSV summary.")


if __name__ == '__main__':
    if config.VERBOSE_LOGGING: print("--- Running Plotting Script: Generating All Composite Figures & CSV Summary ---")
    generate_all_plots_and_summary()