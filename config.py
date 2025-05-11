# config.py (v5 - Harder Obstacles & Path Colors)

# --------------------------------------
# Grid Environment Configuration
# --------------------------------------
GRID_ROWS = 16
GRID_COLS = 16

START_POS = (0, 0)
GOAL_POS = (GRID_ROWS - 1, GRID_COLS - 1) # Adapts to grid size

# New, more challenging obstacle layout for 16x16
OBSTACLES_16x16 = [
    # Outer border-like obstacles to make it a bit more maze-like
    # (r, c) for r in range(GRID_ROWS) for c in [0, GRID_COLS-1] if (r,c) != START_POS and (r,c) != GOAL_POS and not (r==0 and c==0) and not (r==GRID_ROWS-1 and c==GRID_COLS-1)
    # (r, c) for c in range(GRID_COLS) for r in [0, GRID_ROWS-1] if (r,c) != START_POS and (r,c) != GOAL_POS and not (r==0 and c==0) and not (r==GRID_ROWS-1 and c==GRID_COLS-1)
    # Let's define specific obstacles for more control
    # Wall 1 (vertical)
    (r, 3) for r in range(0, GRID_ROWS - 2) if r != 7
] + [
    # Wall 2 (horizontal)
    (7, c) for c in range(1, GRID_COLS - 3)
] + [
    # Wall 3 (vertical)
    (r, GRID_COLS - 4) for r in range(2, GRID_ROWS) if r != 10
] + [
    # Wall 4 (horizontal)
    (12, c) for c in range(GRID_COLS - 2, GRID_COLS - 6, -1) # from right to left
] + [
    # Some scattered blocks
    (2, 8), (2, 9), (4, 1), (5, 1), (4, GRID_COLS-2), (5, GRID_COLS-2),
    (10, 1), (10, 6), (14, 5), (14,6), (14,7)
]
# Ensure start/goal are not obstacles
OBSTACLES_16x16 = list(set(OBSTACLES_16x16)) # Remove duplicates
if START_POS in OBSTACLES_16x16: OBSTACLES_16x16.remove(START_POS)
if GOAL_POS in OBSTACLES_16x16: OBSTACLES_16x16.remove(GOAL_POS)


# New, more challenging obstacle layout for 8x8
OBSTACLES_8x8 = [
    (1,1), (1,2), (1,3), (1,4), (1,5),
    (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), # Wall across
    (5,0), (5,1), (5,2), (5,3), (5,4),
    (6,4), (7,4) # Small passage to goal
]
# Ensure start/goal are not obstacles for 8x8
START_POS_8x8 = (0,0)
GOAL_POS_8x8 = (7,7) # Assuming 8x8 goal
OBSTACLES_8x8 = list(set(OBSTACLES_8x8))
if START_POS_8x8 in OBSTACLES_8x8: OBSTACLES_8x8.remove(START_POS_8x8)
if GOAL_POS_8x8 in OBSTACLES_8x8: OBSTACLES_8x8.remove(GOAL_POS_8x8)

# This will be dynamically chosen in grid_environment.py based on GRID_ROWS
OBSTACLES = [] # Placeholder, will be set in GridEnvironment based on size

REWARD_GOAL = 200 # Increased goal reward for harder maps
REWARD_OBSTACLE = -150 # Increased penalty
REWARD_STEP = -1
REWARD_INVALID_MOVE = -10 # Increased penalty

# --- (Rest of the config remains the same as v4) ---
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 75000 # Adjusted for potentially longer paths

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
NUM_ACTIONS = len(ACTIONS)

LMS_LEARNING_RATE = 0.001
NLMS_LEARNING_RATE = 0.05
NLMS_EPSILON_NORM = 1e-6
SIGN_ERROR_LMS_LEARNING_RATE = 0.0005

RLS_FORGETTING_FACTOR = 0.99
RLS_INIT_P_DIAG_VALUE = 100.0

FEATURE_ENCODING_STRATEGY = 'one_hot_state_action'

NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 750 # Increased for harder maps

ALGORITHMS_TO_RUN = ['LMS', 'NLMS', 'SIGN_ERROR_LMS', 'RLS_LSTD']
RANDOM_SEED = 42

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
CSV_SUMMARY_FILENAME = "experiment_summary.csv"
LOG_FREQUENCY_EPISODES = 100
VERBOSE_LOGGING = True

COLOR_OPTIMAL_PATH = 'cyan'
COLOR_LEARNED_PATH = 'magenta'
PATH_LINEWIDTH = 2
PATH_ALPHA = 0.8

print(f"Configuration file (config.py - v5 Harder Obstacles) loaded.")
print(f"Default NUM_EPISODES: {NUM_EPISODES}, MAX_STEPS_PER_EPISODE: {MAX_STEPS_PER_EPISODE}")
print(f"Default EPSILON_DECAY_STEPS: {EPSILON_DECAY_STEPS}")
