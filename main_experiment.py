# main_experiment.py (v5 - CLI for LRs, passes to RLAgent)

import numpy as np
import os
import time
import random
import argparse
import config # Import your global config
from grid_environment import GridEnvironment
from feature_encoder import FeatureEncoder
from rl_agent import RLAgent

def get_epsilon(current_step, start_eps=config.EPSILON_START, end_eps=config.EPSILON_END, decay_steps=config.EPSILON_DECAY_STEPS):
    if decay_steps <= 0: return end_eps
    fraction = min(1.0, current_step / decay_steps)
    return start_eps - fraction * (start_eps - end_eps)

def run_single_experiment(algorithm_name, current_feature_strategy, cli_args):
    # Override config LRs if provided via CLI
    lms_lr_to_use = cli_args.lms_lr if cli_args.lms_lr is not None else config.LMS_LEARNING_RATE
    nlms_lr_to_use = cli_args.nlms_lr if cli_args.nlms_lr is not None else config.NLMS_LEARNING_RATE
    sign_error_lr_to_use = cli_args.sign_error_lr if cli_args.sign_error_lr is not None else config.SIGN_ERROR_LMS_LEARNING_RATE

    # Update config for this run if CLI overrides are present for grid/episodes
    # This ensures the saved config_snapshot reflects the actual run parameters
    if cli_args.grid_rows is not None: config.GRID_ROWS = cli_args.grid_rows
    if cli_args.grid_cols is not None: config.GRID_COLS = cli_args.grid_cols
    if cli_args.episodes is not None: config.NUM_EPISODES = cli_args.episodes
    
    # Re-initialize goal pos if grid size changed by CLI
    if cli_args.grid_rows is not None or cli_args.grid_cols is not None:
        config.GOAL_POS = (config.GRID_ROWS - 1, config.GRID_COLS - 1)


    print(f"\n--- Starting Experiment: {algorithm_name} | Strategy: {current_feature_strategy} | Grid: {config.GRID_ROWS}x{config.GRID_COLS} ---")
    print(f"Using LRs -> LMS: {lms_lr_to_use}, NLMS: {nlms_lr_to_use}, SignError: {sign_error_lr_to_use}")
    
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    env = GridEnvironment()
    feature_encoder = FeatureEncoder(
        grid_rows=config.GRID_ROWS,
        grid_cols=config.GRID_COLS,
        num_actions=config.NUM_ACTIONS,
        strategy=current_feature_strategy
    )
    
    # Prepare filter_params for RLAgent, including LRs
    filter_params_for_agent = {
        'lms_learning_rate': lms_lr_to_use,
        'nlms_learning_rate': nlms_lr_to_use,
        'nlms_epsilon_norm': config.NLMS_EPSILON_NORM, # from global config
        'sign_error_lms_learning_rate': sign_error_lr_to_use,
        'rls_forgetting_factor': config.RLS_FORGETTING_FACTOR, # from global config
        'rls_init_p_diag_value': config.RLS_INIT_P_DIAG_VALUE  # from global config
    }

    agent = RLAgent(
        feature_encoder_instance=feature_encoder,
        num_actions=config.NUM_ACTIONS,
        adaptive_filter_type=algorithm_name,
        filter_params=filter_params_for_agent # Pass the specific LRs here
    )

    episode_rewards, episode_steps, total_training_steps = [], [], 0
    start_time = time.time()

    for episode in range(config.NUM_EPISODES):
        current_state = env.reset()
        cumulative_reward_this_episode = 0
        ep_steps_count = 0
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Use EPSILON_DECAY_STEPS from config, which might have been updated
            epsilon = get_epsilon(total_training_steps, decay_steps=config.EPSILON_DECAY_STEPS)
            action = agent.choose_action(current_state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.learn(current_state, action, reward, next_state, done)
            current_state = next_state
            cumulative_reward_this_episode += reward
            total_training_steps += 1
            ep_steps_count = step + 1
            if done: break
        
        episode_rewards.append(cumulative_reward_this_episode)
        episode_steps.append(ep_steps_count)

        if (episode + 1) % config.LOG_FREQUENCY_EPISODES == 0 and config.VERBOSE_LOGGING:
            avg_r_log = np.mean(episode_rewards[-config.LOG_FREQUENCY_EPISODES:]) if episode_rewards else 0
            print(f"Algo: {algorithm_name} | Ep: {episode+1}/{config.NUM_EPISODES} | Steps: {ep_steps_count} | Rew: {cumulative_reward_this_episode:.2f} | AvgRew_Last{config.LOG_FREQUENCY_EPISODES}: {avg_r_log:.2f} | Eps: {epsilon:.4f}")

    training_duration = time.time() - start_time
    
    if config.VERBOSE_LOGGING: print("\n--- Post-Training Analysis ---")
    learned_path = agent.get_greedy_path(env)
    optimal_path_bfs = env.get_optimal_path_bfs()
    
    learned_path_len = len(learned_path) -1 if learned_path and len(learned_path) > 0 else config.MAX_STEPS_PER_EPISODE * 2 # Penalty if no path
    optimal_path_len = len(optimal_path_bfs) -1 if optimal_path_bfs and len(optimal_path_bfs) > 0 else -1
    
    path_ratio = -1.0 # Default if optimal path not found or learned path too long
    if optimal_path_len > 0:
        if learned_path_len == config.MAX_STEPS_PER_EPISODE * 2 : # if it timed out on greedy path
             path_ratio = (config.MAX_STEPS_PER_EPISODE * 2) / optimal_path_len # Large ratio
        else:
            path_ratio = learned_path_len / optimal_path_len
    
    if config.VERBOSE_LOGGING:
        print(f"Optimal Path (BFS) Length: {optimal_path_len}")
        print(f"Learned Path Length: {learned_path_len}")
        print(f"Path Length Ratio (Learned/Optimal): {path_ratio:.2f}")

    value_function_grid = agent.get_value_function_grid(config.GRID_ROWS, config.GRID_COLS, env.obstacles)

    num_final_episodes_for_avg = max(1, int(config.NUM_EPISODES * 0.1)) # Avg over last 10%
    avg_final_reward = np.mean(episode_rewards[-num_final_episodes_for_avg:]) if len(episode_rewards) >= num_final_episodes_for_avg else (np.mean(episode_rewards) if episode_rewards else 0)
    avg_final_steps = np.mean(episode_steps[-num_final_episodes_for_avg:]) if len(episode_steps) >= num_final_episodes_for_avg else (np.mean(episode_steps) if episode_steps else 0)
        
    if config.VERBOSE_LOGGING:
        print(f"\n--- Experiment Finished: {algorithm_name} ---")
        print(f"Avg reward over last {num_final_episodes_for_avg} episodes: {avg_final_reward:.2f}")

    # Create a snapshot of relevant config values for this specific run
    # This is crucial if run_all_experiments.ps1 modifies LRs per run
    current_run_config_snapshot = {
        key: value for key, value in config.__dict__.items()
        if not key.startswith('__') and not callable(value) and not isinstance(value, type(config))
    }
    # Explicitly add the LRs used in this run to the snapshot, as they might differ from config file's defaults
    current_run_config_snapshot['LMS_LEARNING_RATE_USED'] = lms_lr_to_use
    current_run_config_snapshot['NLMS_LEARNING_RATE_USED'] = nlms_lr_to_use
    current_run_config_snapshot['SIGN_ERROR_LMS_LEARNING_RATE_USED'] = sign_error_lr_to_use
    current_run_config_snapshot['GRID_ROWS_USED'] = config.GRID_ROWS
    current_run_config_snapshot['GRID_COLS_USED'] = config.GRID_COLS
    current_run_config_snapshot['NUM_EPISODES_USED'] = config.NUM_EPISODES
    current_run_config_snapshot['FEATURE_STRATEGY_USED'] = current_feature_strategy


    results_data = {
        'algorithm': algorithm_name, 
        'feature_strategy': current_feature_strategy,
        'grid_size': f"{config.GRID_ROWS}x{config.GRID_COLS}",
        'num_episodes_configured': config.NUM_EPISODES, # This is what was set for this run
        'max_steps_per_episode_configured': config.MAX_STEPS_PER_EPISODE,
        'total_training_steps_executed': total_training_steps,
        'training_duration_seconds': training_duration,
        'avg_reward_overall': np.mean(episode_rewards) if episode_rewards else 0,
        'avg_steps_overall': np.mean(episode_steps) if episode_steps else 0,
        'avg_reward_final_episodes': avg_final_reward,
        'avg_steps_final_episodes': avg_final_steps,
        'learned_path': learned_path, 
        'optimal_path_bfs': optimal_path_bfs, # Save the actual path
        'learned_path_length': learned_path_len, 
        'optimal_path_length': optimal_path_len,
        'path_length_ratio': path_ratio,
        'value_function_grid': value_function_grid,
        'episode_rewards': episode_rewards, 
        'episode_steps': episode_steps,
        'config_snapshot': current_run_config_snapshot # Save the actual config used
    }
    
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    
    # Make filename more descriptive if LRs are varying
    lr_suffix = ""
    if algorithm_name == "LMS": lr_suffix = f"_lr{lms_lr_to_use:.0e}"
    elif algorithm_name == "NLMS": lr_suffix = f"_lr{nlms_lr_to_use:.0e}" # NLMS also has a base LR
    elif algorithm_name == "SIGN_ERROR_LMS": lr_suffix = f"_lr{sign_error_lr_to_use:.0e}"

    filename_suffix = f"{algorithm_name}_{current_feature_strategy}_{config.GRID_ROWS}x{config.GRID_COLS}_{config.NUM_EPISODES}eps{lr_suffix}.npy"
    result_filename = os.path.join(config.RESULTS_DIR, f"results_{filename_suffix.replace(':', '-')}") # Replace : if any from scientific notation
    
    np.save(result_filename, results_data)
    if config.VERBOSE_LOGGING: print(f"Results saved to {result_filename}")
    return result_filename # Return for plotting script if needed immediately


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiments with adaptive filters.")
    parser.add_argument("--algo", type=str, default='ALL', help="Algorithm (LMS, NLMS, SIGN_ERROR_LMS, RLS_LSTD, ALL).")
    parser.add_argument("--feature_strategy", type=str, default=None, help="Feature strategy (e.g., xy_plus_action, one_hot_state_action). Overrides config default.")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows. Overrides config default.")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid columns. Overrides config default.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes. Overrides config default.")
    
    # New CLI arguments for learning rates
    parser.add_argument("--lms_lr", type=float, default=None, help="LMS learning rate. Overrides config default.")
    parser.add_argument("--nlms_lr", type=float, default=None, help="NLMS base learning rate. Overrides config default.")
    parser.add_argument("--sign_error_lr", type=float, default=None, help="Sign-Error LMS learning rate. Overrides config default.")
    
    args = parser.parse_args()

    # Determine feature strategy: CLI > config default
    effective_feature_strategy = args.feature_strategy if args.feature_strategy is not None else config.FEATURE_ENCODING_STRATEGY
    
    # Store original config values that might be looped over by run_all_experiments
    original_config_num_episodes = config.NUM_EPISODES
    original_config_grid_rows = config.GRID_ROWS
    original_config_grid_cols = config.GRID_COLS
    # Store original LRs in case 'ALL' is run without specific LR overrides from CLI (though PS script will handle this)
    original_lms_lr = config.LMS_LEARNING_RATE
    original_nlms_lr = config.NLMS_LEARNING_RATE
    original_sign_error_lr = config.SIGN_ERROR_LMS_LEARNING_RATE

    if config.VERBOSE_LOGGING:
        print(f"Effective feature strategy for this main_experiment run: {effective_feature_strategy}")
        if args.grid_rows: print(f"Grid Size overridden by CLI: {args.grid_rows}x{args.grid_cols}")
        if args.episodes: print(f"Episodes overridden by CLI: {args.episodes}")
        if args.lms_lr: print(f"LMS LR overridden by CLI: {args.lms_lr}")
        if args.nlms_lr: print(f"NLMS LR overridden by CLI: {args.nlms_lr}")
        if args.sign_error_lr: print(f"SignError LR overridden by CLI: {args.sign_error_lr}")


    if args.algo.upper() == 'ALL':
        print("Running experiments for ALL configured algorithms (as defined in config.ALGORITHMS_TO_RUN)...")
        # Note: If run_all_experiments.ps1 calls this with 'ALL', it should ideally loop itself
        # and call main_experiment.py for each specific algo with its specific params.
        # This 'ALL' block here is more for direct `python main_experiment.py --algo ALL` usage.
        for algo_name in config.ALGORITHMS_TO_RUN:
            # Reset global config for each run if CLI args were for a single run context
            config.NUM_EPISODES = args.episodes if args.episodes is not None else original_config_num_episodes
            config.GRID_ROWS = args.grid_rows if args.grid_rows is not None else original_config_grid_rows
            config.GRID_COLS = args.grid_cols if args.grid_cols is not None else original_config_grid_cols
            # For LRs, use CLI if provided for this 'ALL' call, else use original config defaults
            temp_args_for_loop = argparse.Namespace(**vars(args)) # copy args
            if args.lms_lr is None: temp_args_for_loop.lms_lr = original_lms_lr
            if args.nlms_lr is None: temp_args_for_loop.nlms_lr = original_nlms_lr
            if args.sign_error_lr is None: temp_args_for_loop.sign_error_lr = original_sign_error_lr
            
            run_single_experiment(algo_name, effective_feature_strategy, temp_args_for_loop)
    elif args.algo.upper() in config.ALGORITHMS_TO_RUN:
        # When called from PowerShell, args will contain the specific LRs for this algo run
        run_single_experiment(args.algo.upper(), effective_feature_strategy, args)
    else:
        print(f"Error: Algorithm '{args.algo}' is not recognized or not in config.ALGORITHMS_TO_RUN.")
    
    if config.VERBOSE_LOGGING: print("\nmain_experiment.py script finished.")