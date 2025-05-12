# config.py (v8 - Reverted to Original Successful Settings for one_hot_state_action)

import numpy as np

# --------------------------------------
# Grid Environment Configuration
# --------------------------------------
GRID_ROWS = 16 # Default, can be overridden by run_all_experiments.ps1
GRID_COLS = 16 # Default, can be overridden by run_all_experiments.ps1

START_POS = (0, 0)
GOAL_POS = (GRID_ROWS - 1, GRID_COLS - 1) # Adapts to grid size

# Obstacle layouts from your "v5 - Harder Obstacles" config, (used in original successful run)
OBSTACLES_16x16 = [
    (r, 3) for r in range(0, GRID_ROWS - 2) if r != 7
] + [
    (7, c) for c in range(1, GRID_COLS - 3)
] + [
    (r, GRID_COLS - 4) for r in range(2, GRID_ROWS) if r != 10
] + [
    (12, c) for c in range(GRID_COLS - 2, GRID_COLS - 6, -1) # from right to left
] + [
    (2, 8), (2, 9), (4, 1), (5, 1), (4, GRID_COLS-2), (5, GRID_COLS-2),
    (10, 1), (10, 6), (14, 5), (14,6), (14,7)
]
OBSTACLES_16x16 = list(set(OBSTACLES_16x16)) # Remove duplicates
if START_POS in OBSTACLES_16x16: OBSTACLES_16x16.remove(START_POS)
if GOAL_POS in OBSTACLES_16x16: OBSTACLES_16x16.remove(GOAL_POS)

OBSTACLES_8x8 = [
    (1,1), (1,2), (1,3), (1,4), (1,5),
    (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), # Wall across
    (5,0), (5,1), (5,2), (5,3), (5,4),
    (6,4), (7,4) # Small passage to goal
]
START_POS_8x8 = (0,0)
GOAL_POS_8x8 = (7,7)
OBSTACLES_8x8 = list(set(OBSTACLES_8x8))
if START_POS_8x8 in OBSTACLES_8x8: OBSTACLES_8x8.remove(START_POS_8x8)
if GOAL_POS_8x8 in OBSTACLES_8x8: OBSTACLES_8x8.remove(GOAL_POS_8x8)

OBSTACLES = [] # Placeholder, will be set in GridEnvironment based on size

# Rewards (from main.tex)
REWARD_GOAL = 200
REWARD_OBSTACLE = -150
REWARD_STEP = -1
REWARD_INVALID_MOVE = -10 # Original value from main.tex
REWARD_SHAPING_FACTOR = 0.0 # IMPORTANT: Turn OFF reward shaping

# RL Parameters (from main.tex)
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 75000 # Original value from main.tex

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
NUM_ACTIONS = len(ACTIONS)

# Adaptive Filter Parameters (Original successful values from main.tex)
LMS_LEARNING_RATE = 0.001
NLMS_LEARNING_RATE = 0.05
NLMS_EPSILON_NORM = 1e-6
SIGN_ERROR_LMS_LEARNING_RATE = 0.0005

RLS_FORGETTING_FACTOR = 0.99
RLS_INIT_P_DIAG_VALUE = 100.0

# Default feature encoding strategy
FEATURE_ENCODING_STRATEGY = 'one_hot_state_action' # Reverted to original

# Experiment Runner Defaults (can be overridden by CLI in main_experiment.py or run_all_experiments.ps1)
# These will be set by run_all_experiments.ps1 based on its logic
NUM_EPISODES = 5000 # Default for larger grid
MAX_STEPS_PER_EPISODE = 750

ALGORITHMS_TO_RUN = ['LMS', 'NLMS', 'SIGN_ERROR_LMS', 'RLS_LSTD']
RANDOM_SEED = 42

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
CSV_SUMMARY_FILENAME = "experiment_summary_original_config.csv" # New name for these reverted results

LOG_FREQUENCY_EPISODES = 100
VERBOSE_LOGGING = True # Set to False to reduce console output

COLOR_OPTIMAL_PATH = 'cyan'
COLOR_LEARNED_PATH = 'magenta'
PATH_LINEWIDTH = 2
PATH_ALPHA = 0.8

print(f"Configuration file (config.py - v8 Reverted to Original Successful Settings) loaded.")
print(f"FEATURE_ENCODING_STRATEGY: {FEATURE_ENCODING_STRATEGY}")
print(f"EPSILON_DECAY_STEPS: {EPSILON_DECAY_STEPS}")
print(f"REWARD_INVALID_MOVE: {REWARD_INVALID_MOVE}")
print(f"REWARD_SHAPING_FACTOR: {REWARD_SHAPING_FACTOR}")
print(f"LMS_LR: {LMS_LEARNING_RATE}, NLMS_LR: {NLMS_LEARNING_RATE}, SIGN_ERROR_LR: {SIGN_ERROR_LMS_LEARNING_RATE}")
print(f"RLS Lambda: {RLS_FORGETTING_FACTOR}, RLS P_init: {RLS_INIT_P_DIAG_VALUE}")