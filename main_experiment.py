# main_experiment.py (v4)

import numpy as np
import os
import time
import random
import argparse
import config
from grid_environment import GridEnvironment
from feature_encoder import FeatureEncoder
from rl_agent import RLAgent

def get_epsilon(current_step, start_eps=config.EPSILON_START, end_eps=config.EPSILON_END, decay_steps=config.EPSILON_DECAY_STEPS):
    if decay_steps <= 0: return end_eps
    fraction = min(1.0, current_step / decay_steps)
    return start_eps - fraction * (start_eps - end_eps)

def run_single_experiment(algorithm_name, current_feature_strategy):
    print(f"\n--- Starting Experiment: {algorithm_name} | Strategy: {current_feature_strategy} | Grid: {config.GRID_ROWS}x{config.GRID_COLS} ---")
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    env = GridEnvironment() # Uses GRID_ROWS, GRID_COLS from config for its internal setup
    feature_encoder = FeatureEncoder(
        grid_rows=config.GRID_ROWS, # Pass current config grid size
        grid_cols=config.GRID_COLS,
        num_actions=config.NUM_ACTIONS,
        strategy=current_feature_strategy
    )
    agent = RLAgent(
        feature_encoder_instance=feature_encoder,
        num_actions=config.NUM_ACTIONS,
        adaptive_filter_type=algorithm_name
    )

    episode_rewards, episode_steps, total_training_steps = [], [], 0
    start_time = time.time()

    for episode in range(config.NUM_EPISODES):
        current_state = env.reset()
        cumulative_reward_this_episode = 0
        ep_steps_count = 0
        for step in range(config.MAX_STEPS_PER_EPISODE):
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

        if (episode + 1) % config.LOG_FREQUENCY_EPISODES == 0:
            avg_r = np.mean(episode_rewards[-config.LOG_FREQUENCY_EPISODES:])
            print(f"Algo: {algorithm_name} | Ep: {episode+1}/{config.NUM_EPISODES} | Steps: {ep_steps_count} | Rew: {cumulative_reward_this_episode:.2f} | AvgRew: {avg_r:.2f} | Eps: {epsilon:.4f}")

    training_duration = time.time() - start_time
    
    # Post-training analysis
    print("\n--- Post-Training Analysis ---")
    learned_path = agent.get_greedy_path(env) # Pass the same env instance
    optimal_path_bfs = env.get_optimal_path_bfs()
    
    learned_path_len = len(learned_path) -1 if learned_path else -1
    optimal_path_len = len(optimal_path_bfs) -1 if optimal_path_bfs else -1
    path_ratio = (learned_path_len / optimal_path_len) if optimal_path_len > 0 and learned_path_len > -1 else -1

    print(f"Optimal Path (BFS) Length: {optimal_path_len}")
    print(f"Learned Path Length: {learned_path_len}")
    print(f"Path Length Ratio (Learned/Optimal): {path_ratio:.2f}")

    value_function_grid = agent.get_value_function_grid(config.GRID_ROWS, config.GRID_COLS, env.obstacles)

    # Summary stats
    num_final_episodes_for_avg = max(1, int(config.NUM_EPISODES * 0.2))
    avg_final_reward = np.mean(episode_rewards[-num_final_episodes_for_avg:]) if len(episode_rewards) >= num_final_episodes_for_avg else (np.mean(episode_rewards) if episode_rewards else 0)
    avg_final_steps = np.mean(episode_steps[-num_final_episodes_for_avg:]) if len(episode_steps) >= num_final_episodes_for_avg else (np.mean(episode_steps) if episode_steps else 0)
        
    print(f"\n--- Experiment Finished: {algorithm_name} ---")
    # ... (other print statements from v3) ...
    print(f"Avg reward over last {num_final_episodes_for_avg} episodes: {avg_final_reward:.2f}")

    results_data = {
        'algorithm': algorithm_name, 'feature_strategy': current_feature_strategy,
        'grid_size': f"{config.GRID_ROWS}x{config.GRID_COLS}",
        'num_episodes_configured': config.NUM_EPISODES,
        'max_steps_per_episode_configured': config.MAX_STEPS_PER_EPISODE,
        'total_training_steps_executed': total_training_steps,
        'training_duration_seconds': training_duration,
        'avg_reward_overall': np.mean(episode_rewards) if episode_rewards else 0,
        'avg_steps_overall': np.mean(episode_steps) if episode_steps else 0,
        'avg_reward_final_episodes': avg_final_reward,
        'avg_steps_final_episodes': avg_final_steps,
        'num_final_episodes_for_avg': num_final_episodes_for_avg,
        'learned_path': learned_path, 'optimal_path_bfs': optimal_path_bfs,
        'learned_path_length': learned_path_len, 'optimal_path_length': optimal_path_len,
        'path_length_ratio': path_ratio,
        'value_function_grid': value_function_grid,
        'episode_rewards': episode_rewards, 'episode_steps': episode_steps,
        'config_snapshot': {key: getattr(config,key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config,key))}
    }
    
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    filename_suffix = f"{algorithm_name}_{current_feature_strategy}_{config.GRID_ROWS}x{config.GRID_COLS}_{config.NUM_EPISODES}eps.npy"
    result_filename = os.path.join(config.RESULTS_DIR, f"results_{filename_suffix}")
    np.save(result_filename, results_data)
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiments.")
    parser.add_argument("--algo", type=str, choices=config.ALGORITHMS_TO_RUN + ['ALL'], default='ALL', help="Algorithm.")
    parser.add_argument("--feature_strategy", type=str, default=config.FEATURE_ENCODING_STRATEGY, help="Feature strategy.")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows.")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid columns.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes.")
    args = parser.parse_args()

    original_config_num_episodes = config.NUM_EPISODES # Store original for resetting if looping

    if args.grid_rows is not None: config.GRID_ROWS = args.grid_rows
    if args.grid_cols is not None: config.GRID_COLS = args.grid_cols
    if args.grid_rows is not None or args.grid_cols is not None:
        config.START_POS = (0,0) # Reset start if grid size changes
        config.GOAL_POS = (config.GRID_ROWS - 1, config.GRID_COLS - 1)
        current_obstacles = config.OBSTACLES # Use the list from config.py
        config.OBSTACLES = [(r, c) for r,c in current_obstacles if r < config.GRID_ROWS and c < config.GRID_COLS]
        if config.GOAL_POS in config.OBSTACLES: config.OBSTACLES.remove(config.GOAL_POS)
        if config.START_POS in config.OBSTACLES: config.OBSTACLES.remove(config.START_POS)

    if args.episodes is not None: config.NUM_EPISODES = args.episodes
    
    effective_feature_strategy = args.feature_strategy
    if args.feature_strategy == 'simple_coords':
        # print(f"Warning: 'simple_coords' selected. For Q(s,a) with current RLAgent, "
        #       f"this implies features are for V(s) or FeatureEncoder needs to create phi(s,a). "
        #       f"Using 'one_hot_state_action' as a compatible default for this run to ensure Q(s,a) structure matches.")
        effective_feature_strategy = 'one_hot_state_action' # Ensure compatibility

    print(f"Effective feature strategy for run: {effective_feature_strategy}")
    print(f"Grid Size for run: {config.GRID_ROWS}x{config.GRID_COLS}, Start: {config.START_POS}, Goal: {config.GOAL_POS}, Episodes: {config.NUM_EPISODES}")

    if args.algo.upper() == 'ALL':
        print("Running experiments for ALL configured algorithms...")
        for algo_name in config.ALGORITHMS_TO_RUN:
            # Reset num_episodes from CLI if it was set for a specific run, or use config default
            if args.episodes is not None:
                 config.NUM_EPISODES = args.episodes
            else: # Reset to original config value if looping through ALL and no CLI override for episodes
                 config.NUM_EPISODES = original_config_num_episodes

            run_single_experiment(algo_name, effective_feature_strategy)
    elif args.algo.upper() in config.ALGORITHMS_TO_RUN:
        print(f"Running experiment for specified algorithm: {args.algo.upper()}")
        run_single_experiment(args.algo.upper(), effective_feature_strategy)
    else:
        print(f"Error: Algorithm '{args.algo}' is not recognized.")
    print("\nAll specified experiments complete.")
