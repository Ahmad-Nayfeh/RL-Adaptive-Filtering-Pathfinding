# plotting.py (v7)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
import csv
import config

def plot_grid_map_with_paths(grid_rows, grid_cols, start_pos, goal_pos, obstacles,
                             optimal_path=None, learned_path=None,
                             title="Grid Map with Paths", filename="grid_map_with_paths.png"):
    """
    Generates and saves an image of the grid map with optional paths.
    Improved grid line appearance.
    """
    fig, ax = plt.subplots(figsize=(max(6, grid_cols/1.5), max(6, grid_rows/1.5))) # Adjusted figsize
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5) # Invert y-axis for (0,0) at top-left

    # Cell appearance
    cell_edge_color = 'lightgray'
    cell_linewidth = 0.5

    cmap = {'empty': 'white', 'obstacle': '#404040', 'start': 'green', 'goal': 'red'}

    # Draw cells (background and border)
    for r_idx in range(grid_rows):
        for c_idx in range(grid_cols):
            color = cmap['empty']
            if (r_idx, c_idx) in obstacles:
                color = cmap['obstacle']
            elif (r_idx, c_idx) == start_pos:
                color = cmap['start']
            elif (r_idx, c_idx) == goal_pos:
                color = cmap['goal']
            
            rect = mpatches.Rectangle((c_idx - 0.5, r_idx - 0.5), 1, 1, 
                                      facecolor=color, edgecolor=cell_edge_color, linewidth=cell_linewidth)
            ax.add_patch(rect)
            # Add S and G text inside start/goal cells
            if (r_idx, c_idx) == start_pos:
                ax.text(c_idx, r_idx, 'S', ha='center', va='center', color='white', fontsize=8, weight='bold')
            if (r_idx, c_idx) == goal_pos:
                ax.text(c_idx, r_idx, 'G', ha='center', va='center', color='white', fontsize=8, weight='bold')


    # Plot paths on top
    path_handles = []
    if optimal_path and len(optimal_path) > 1:
        op_r, op_c = zip(*optimal_path)
        ax.plot(op_c, op_r, color=config.COLOR_OPTIMAL_PATH, linewidth=config.PATH_LINEWIDTH, 
                alpha=config.PATH_ALPHA, marker='.', markersize=4, zorder=3)
        path_handles.append(mpatches.Patch(color=config.COLOR_OPTIMAL_PATH, 
                                           label=f'Optimal Path ({len(optimal_path)-1} steps)'))
    
    if learned_path and len(learned_path) > 1:
        lp_r, lp_c = zip(*learned_path)
        ax.plot(lp_c, lp_r, color=config.COLOR_LEARNED_PATH, linewidth=config.PATH_LINEWIDTH, 
                linestyle='--', alpha=config.PATH_ALPHA, marker='.', markersize=4, zorder=4)
        path_handles.append(mpatches.Patch(color=config.COLOR_LEARNED_PATH, 
                                           label=f'Learned Path ({len(learned_path)-1} steps)'))

    # Remove axis ticks and labels for a cleaner map
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticklabels([]) # Not needed if ticks are removed
    # ax.set_yticklabels([]) # Not needed if ticks are removed

    if path_handles:
        ax.legend(handles=path_handles, fontsize='small', loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2) # Legend below map
    
    plt.title(title, fontsize=12)
    plt.box(False) # Remove plot box/frame
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for legend and title

    if not os.path.exists(config.PLOTS_DIR): os.makedirs(config.PLOTS_DIR)
    full_save_path = os.path.join(config.PLOTS_DIR, filename)
    plt.savefig(full_save_path, dpi=200) # Increased DPI for better quality
    print(f"Path map saved to {full_save_path}")
    plt.close(fig)

# --- plot_value_heatmap, plot_learning_curves, plot_summary_statistics, generate_csv_summary (largely same as v6) ---
# Small adjustments might be needed in generate_all_plots_and_summary to correctly pass S,G,O to plot_grid_map_with_paths

def plot_value_heatmap(v_grid, grid_rows, grid_cols, obstacles, title="Value Function Heatmap", filename="value_heatmap.png"):
    if v_grid is None: print(f"Skipping heatmap for {title} as value_function_grid is None."); return
    fig, ax = plt.subplots(figsize=(max(7, grid_cols/1.5), max(6, grid_rows/1.5))) # Adjusted figsize
    masked_v_grid = np.ma.array(v_grid, mask=np.isnan(v_grid))
    min_val, max_val = np.nanmin(v_grid), np.nanmax(v_grid)
    if np.all(np.isnan(v_grid)): min_val, max_val = 0,1 # Handle all NaN case
    elif min_val == max_val: min_val -= 0.5; max_val += 0.5
    if min_val == 0 and max_val == 0: max_val = 1
    cmap = plt.cm.viridis; cmap.set_bad(color='#505050')
    cax = ax.imshow(masked_v_grid, cmap=cmap, interpolation='nearest', vmin=min_val, vmax=max_val)
    ax.set_xticks(np.arange(grid_cols)); ax.set_yticks(np.arange(grid_rows))
    ax.set_xticklabels(np.arange(grid_cols)); ax.set_yticklabels(np.arange(grid_rows))
    ax.tick_params(axis='both', which='major', labelsize=8)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if not np.isnan(v_grid[r, c]):
                val = v_grid[r,c]; norm_val = (val - min_val) / (max_val - min_val + 1e-9) # Normalize for color decision
                text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=max(4, 10 - grid_cols // 2)) # Dynamic font size
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="State Value V(s)")
    plt.title(title, fontsize=12); plt.tight_layout()
    if not os.path.exists(config.PLOTS_DIR): os.makedirs(config.PLOTS_DIR)
    plt.savefig(os.path.join(config.PLOTS_DIR, filename), dpi=200); print(f"Value heatmap: {filename}"); plt.close(fig)


def plot_learning_curves(results_data_list, title_suffix="Learning Curves", save_filename="learning_curves.png", show_raw=False): # (Same as v6)
    if not isinstance(results_data_list, list): results_data_list = [results_data_list]
    plt.figure(figsize=(14, 9))
    for data in results_data_list:
        algo, feat, grid_s = data.get('algorithm','Unk'), data.get('feature_strategy',''), data.get('grid_size','')
        label = f"{algo}_{feat.replace('_',' ').title()} ({grid_s})" if feat else f"{algo} ({grid_s})"
        rewards = data.get('episode_rewards', [])
        if not rewards: print(f"W: No rewards for {label}"); continue
        if show_raw: plt.plot(rewards, label=f'{label} (Raw)', alpha=0.3)
        n_eps = len(rewards)
        win = max(1, min(config.LOG_FREQUENCY_EPISODES, n_eps // 10 if n_eps > 20 else (n_eps // 2 if n_eps > 3 else 1) ))
        if n_eps >= win and win > 0:
            avg_r = np.convolve(rewards, np.ones(win)/win, mode='valid')
            plt.plot(np.arange(win-1, n_eps), avg_r, label=f'{label} (MovAvg {win}ep)')
        else: plt.plot(rewards, label=f'{label} (Raw - FewPts)')
    plt.title(f"Agent Performance: {title_suffix}"); plt.xlabel("Episode"); plt.ylabel("Cumulative Reward")
    plt.legend(fontsize='medium', loc='best'); plt.grid(True, alpha=0.7); plt.tight_layout()
    if not os.path.exists(config.PLOTS_DIR): os.makedirs(config.PLOTS_DIR)
    plt.savefig(os.path.join(config.PLOTS_DIR, save_filename), dpi=150); print(f"LC plot: {save_filename}"); plt.close()

def plot_summary_statistics(results_data_list, metrics_to_plot=None, base_filename="summary_stats"): # (Same as v6)
    if not results_data_list: return
    if metrics_to_plot is None: metrics_to_plot = ['training_duration_seconds', 'avg_reward_final_episodes', 'avg_steps_final_episodes', 'path_length_ratio', 'total_training_steps_executed']
    labels = [f"{d.get('algorithm','N/A')}\n({d.get('grid_size','N/A')}, {d.get('num_episodes_configured','N/A')}eps)" for d in results_data_list]
    for metric in metrics_to_plot:
        values = [d.get(metric, 0) for d in results_data_list]
        if not any(v != 0 for v in values) and not metric.startswith("avg_") and metric != 'path_length_ratio': print(f"Skipping {metric}, all zeros/missing."); continue
        plt.figure(figsize=(max(10, len(labels)*1.8),8)); bars = plt.bar(labels, values, color=plt.cm.viridis(np.linspace(0,1,len(labels))))
        plt.ylabel(metric.replace('_',' ').title()); plt.title(f"Comparison: {metric.replace('_',' ').title()}",fontsize=14)
        plt.xticks(rotation=20,ha="right",fontsize='small'); plt.grid(True,axis='y',alpha=0.7); plt.tight_layout()
        for bar in bars:
            y = bar.get_height(); offset = 0.01 * max(values if any(v!=0 for v in values) else [1],default=1)
            if y < (0.05 * max(values if any(v!=0 for v in values) else [1],default=1)): offset += abs(y)
            plt.text(bar.get_x()+bar.get_width()/2.,y+offset,f'{y:.2f}',ha='center',va='bottom',fontsize='x-small')
        if not os.path.exists(config.PLOTS_DIR): os.makedirs(config.PLOTS_DIR)
        fname = os.path.join(config.PLOTS_DIR,f"{base_filename}_{metric}.png"); plt.savefig(fname,dpi=150); print(f"Summary plot: {fname}"); plt.close()

def generate_csv_summary(results_data_list): # (Same as v6)
    if not results_data_list: return
    fieldnames = [
        'Algorithm', 'Feature_Strategy', 'Grid_Size', 'Num_Episodes_Configured', 'Max_Steps_Per_Episode_Configured',
        'Total_Training_Steps_Executed', 'Training_Duration_Seconds',
        'Avg_Reward_Overall', 'Avg_Steps_Overall', 'Avg_Reward_Final_Episodes', 'Avg_Steps_Final_Episodes',
        'Learned_Path_Length', 'Optimal_Path_Length', 'Path_Length_Ratio',
        'LMS_LR', 'NLMS_LR', 'SignError_LR', 'RLS_Lambda', 'RLS_P_Init', 'Gamma', 'Random_Seed'
    ]
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    csv_path = os.path.join(config.RESULTS_DIR, config.CSV_SUMMARY_FILENAME)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for data in results_data_list:
            cfg = data.get('config_snapshot',{})
            row = {k: data.get(k.lower(), data.get(k)) for k in fieldnames if data.get(k.lower(), data.get(k)) is not None}
            row.update({
                'Algorithm': data.get('algorithm'), 'Feature_Strategy': data.get('feature_strategy'), 'Grid_Size': data.get('grid_size'),
                'Num_Episodes_Configured': data.get('num_episodes_configured'),
                'Max_Steps_Per_Episode_Configured': data.get('max_steps_per_episode_configured'),
                'Total_Training_Steps_Executed': data.get('total_training_steps_executed'),
                'Training_Duration_Seconds': f"{data.get('training_duration_seconds',0):.2f}",
                'Avg_Reward_Overall': f"{data.get('avg_reward_overall',0):.2f}",
                'Avg_Steps_Overall': f"{data.get('avg_steps_overall',0):.2f}",
                'Avg_Reward_Final_Episodes': f"{data.get('avg_reward_final_episodes',0):.2f}",
                'Avg_Steps_Final_Episodes': f"{data.get('avg_steps_final_episodes',0):.2f}",
                'Learned_Path_Length': data.get('learned_path_length', -1),
                'Optimal_Path_Length': data.get('optimal_path_length', -1),
                'Path_Length_Ratio': f"{data.get('path_length_ratio', -1):.2f}",
                'LMS_LR': cfg.get('LMS_LEARNING_RATE'), 'NLMS_LR': cfg.get('NLMS_LEARNING_RATE'),
                'SignError_LR': cfg.get('SIGN_ERROR_LMS_LEARNING_RATE'), 'RLS_Lambda': cfg.get('RLS_FORGETTING_FACTOR'),
                'RLS_P_Init': cfg.get('RLS_INIT_P_DIAG_VALUE'), 'Gamma': cfg.get('GAMMA'), 'Random_Seed': cfg.get('RANDOM_SEED')
            })
            writer.writerow(row)
    print(f"CSV summary saved to {csv_path}")

def generate_all_plots_and_summary():
    result_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "results_*.npy")))
    if not result_files: print(f"No result files found in {config.RESULTS_DIR}"); return
    print(f"Found {len(result_files)} result files.")
    all_data = []

    for fp in result_files:
        try:
            data = np.load(fp, allow_pickle=True).item(); all_data.append(data)
            cfg_snap = data.get('config_snapshot', {})
            grid_s = data.get('grid_size', 'UnkGrid')
            algo = data.get('algorithm','UnkAlgo')
            feat = data.get('feature_strategy','UnkFeat')
            eps_run = data.get('num_episodes_configured','UnkEps')

            # Plot Grid Map with Paths for EACH result file
            if grid_s != 'UnkGrid':
                try:
                    rows, cols = map(int, grid_s.split('x'))
                    # Get S, G, O from the config_snapshot of this specific run
                    # If not in snapshot, use current config as fallback (dynamic based on size)
                    if rows == 16:
                        run_obstacles = config.OBSTACLES_16x16
                        start_pos = cfg_snap.get('START_POS', config.START_POS) # Default to global config if not in snap
                        goal_pos = cfg_snap.get('GOAL_POS', config.GOAL_POS)
                    elif rows == 8:
                        run_obstacles = config.OBSTACLES_8x8
                        start_pos = cfg_snap.get('START_POS_8x8', config.START_POS_8x8)
                        goal_pos = cfg_snap.get('GOAL_POS_8x8', config.GOAL_POS_8x8)
                    else: # Fallback for other sizes
                        run_obstacles = []
                        start_pos = (0,0)
                        goal_pos = (rows-1, cols-1)
                    
                    # Final check to ensure S/G are not obstacles for this specific map instance
                    final_obstacles = set(run_obstacles)
                    if start_pos in final_obstacles: final_obstacles.remove(start_pos)
                    if goal_pos in final_obstacles: final_obstacles.remove(goal_pos)

                    map_title = f"{algo} Path ({grid_s}, {eps_run}eps)"
                    map_fname = f"map_path_{algo}_{feat}_{grid_s}_{eps_run}eps.png"
                    print(f"\nPlotting map with paths for {fp.split(os.sep)[-1]}")
                    plot_grid_map_with_paths(rows, cols, start_pos, goal_pos, list(final_obstacles),
                                             data.get('optimal_path_bfs'), data.get('learned_path'),
                                             title=map_title, filename=map_fname)
                    
                    v_grid = data.get('value_function_grid')
                    if v_grid is not None:
                        heatmap_title = f"{algo} V-func ({grid_s}, {eps_run}eps)"
                        heatmap_fname = f"heatmap_V_{algo}_{feat}_{grid_s}_{eps_run}eps.png"
                        print(f"Plotting V-function heatmap for {fp.split(os.sep)[-1]}")
                        plot_value_heatmap(v_grid, rows, cols, list(final_obstacles), title=heatmap_title, filename=heatmap_fname)
                except Exception as e: print(f"Error plotting map/heatmap for {fp}: {e}")

            # Individual Learning Curve
            lc_title = f"{algo} ({feat.replace('_',' ').title()}, {grid_s}, {eps_run}eps)"
            lc_fname = f"plot_curve_{algo}_{feat}_{grid_s}_{eps_run}eps.png"
            print(f"Plotting individual LC for {fp.split(os.sep)[-1]}")
            plot_learning_curves([data], title_suffix=lc_title, save_filename=lc_fname, show_raw=True)

        except Exception as e: print(f"Error loading/processing file {fp}: {e}")
    
    if not all_data: print("No data loaded. Exiting plot generation."); return
    if len(all_data) > 1:
        print("\nGenerating combined learning curve..."); plot_learning_curves(all_data, title_suffix="All Runs Comparison", save_filename="A_COMBINED_learning_curves.png")
    print("\nGenerating summary statistics plots..."); plot_summary_statistics(all_data)
    print("\nGenerating CSV summary..."); generate_csv_summary(all_data)

if __name__ == "__main__":
    print("--- Auto-Generating All Plots, Maps, and CSV Summary from Results Directory ---")
    generate_all_plots_and_summary()
