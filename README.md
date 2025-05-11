# Comparative Study of Adaptive Filtering Algorithms for Reinforcement Learning in Grid-World Pathfinding

## 1. Introduction

This project implements and evaluates a Reinforcement Learning (RL) agent designed to navigate a grid-world environment with static obstacles. The core objective is to find an optimal path from a defined start position to a goal position. The primary innovation lies in the agent's learning mechanism: its value function, which guides its decision-making, is updated using various adaptive filtering algorithms. This study specifically compares the performance (e.g., learning speed, path optimality, computational aspects) of different algorithms from the Least Mean Squares (LMS) and Recursive Least Squares (RLS) families when applied to this RL task.

The agent employs Q-learning with linear function approximation. The Temporal Difference (TD) error, derived from the Q-learning updates, serves as the error signal for the adaptive filters, which then adjust the weights of the linear value function approximator:

$Q(s,a;\mathbf{w}) = \mathbf{w}^{\mathsf{T}}\phi(s,a)$

where $\mathbf{w}$ are the weights and $\phi(s,a)$ is the feature vector for a given state-action pair.

## 2. Core Concepts

* **Reinforcement Learning (RL):** An area of machine learning where an agent learns to make a sequence of decisions by trial and error in an environment to maximize a cumulative reward.
* **Q-Learning:** A model-free RL algorithm that learns a state-action value function (Q-function), which represents the expected utility of taking a given action in a given state and following an optimal policy thereafter.
* **Value Function Approximation (VFA):** Used when the state-action space is too large for a tabular representation. A function (in this case, linear) is used to approximate the Q-values, parameterized by a set of weights.
* **Adaptive Filtering:** A class of digital signal processing algorithms that iteratively adjust the parameters of a filter to approach an optimal desired output. In this project, they are used to update the weights of the VFA based on the TD error.

## 3. Algorithms Implemented & Compared

The following adaptive filtering algorithms are implemented for updating the Q-function weights:

* **LMS Family:**
    * Standard Least Mean Squares (LMS)
    * Normalized Least Mean Squares (NLMS)
    * Sign-Error LMS
* **RLS Family:**
    * Recursive Least Squares (RLS), applied in the context of Least Squares Temporal Difference (LSTD-Q with $\lambda=0$).

## 4. Project Structure

The project is organized into several Python scripts:

* `config.py`: Contains all global configurations, hyperparameters for the environment, RL agent, adaptive filters, and experiment settings.
* `grid_environment.py`: Defines the `GridEnvironment` class, managing the grid, obstacles, agent movement, and reward structure. Includes a Breadth-First Search (BFS) method for finding the optimal path.
* `feature_encoder.py`: Defines the `FeatureEncoder` class, responsible for converting environment states (and actions) into numerical feature vectors (e.g., using `one_hot_state_action` encoding).
* `adaptive_filters.py`: Contains implementations of the LMS, NLMS, Sign-Error LMS, and RLS update rules.
* `rl_agent.py`: Defines the `RLAgent` class, which implements the Q-learning logic, action selection (epsilon-greedy), and interfaces with the feature encoder and adaptive filters to learn. Includes methods for tracing the learned path and calculating the V-function.
* `main_experiment.py`: The main script for running training experiments. It initializes the environment and agent, runs the training loop, and saves results (including paths and V-functions). It supports command-line arguments for selecting algorithms, grid sizes, and episode counts.
* `plotting.py`: Contains functions to generate various plots from the saved experiment results, including learning curves, grid maps with optimal/learned paths, V-function heatmaps, and summary statistic bar charts. It also generates a CSV summary of all experiments.
* `run_all_experiments.ps1`: A PowerShell script to automate running a batch of predefined experiments (e.g., different algorithms on different grid sizes) and subsequently generate all plots and the summary CSV.

## 5. Prerequisites & Setup

* Python 3.x
* NumPy: `pip install numpy`
* Matplotlib: `pip install matplotlib`

No other special setup is required beyond having Python and these libraries installed.

## 6. Running Experiments

### 6.1. Automated Batch Experiments (Recommended)

The most convenient way to run a comprehensive set of experiments and generate all outputs is using the PowerShell script:

```powershell
.\run_all_experiments.ps1
````

- **Before running:**
    
    - Open `run_all_experiments.ps1` and configure the desired episode counts (`$episodes_large_grid`, `$episodes_small_grid`) and grid sizes (`$grid_size_large`, `$grid_size_small`) for the experimental sets.
    - Ensure Python is in your system's PATH or update `$PythonExecutable` in the script.
    - If it's your first time running PowerShell scripts, you might need to adjust the execution policy (run PowerShell as Administrator): `Set-ExecutionPolicy RemoteSigned`
- **What it does:**
    
    1. Runs experiments for LMS, NLMS, Sign-Error LMS on a 16x16 grid.
    2. Runs experiments for LMS, NLMS, Sign-Error LMS, and RLS_LSTD on a smaller grid (e.g., 8x8).
    3. Saves all numerical results (episode rewards, paths, V-functions, timings, etc.) as `.npy` files in the `results/` directory.
    4. Calls `plotting.py` to automatically generate:
        - Individual learning curve plots for each run.
        - A combined learning curve plot comparing all runs.
        - Grid maps showing optimal and learned paths for each run.
        - V-function heatmaps for each run.
        - Bar charts comparing summary statistics (e.g., final reward, training time, path length ratio).
        - A `experiment_summary.csv` file in the `results/` directory with tabulated outcomes.
        - All plots are saved in the `plots/` directory.

### 6.2. Individual Experiments

You can also run individual experiments using `main_experiment.py` with command-line arguments:

PowerShell

```
python .\main_experiment.py --algo <ALGORITHM_NAME> --grid_rows <ROWS> --grid_cols <COLS> --episodes <NUM_EPISODES> --feature_strategy <STRATEGY>
```

- **`<ALGORITHM_NAME>`:** One of `LMS`, `NLMS`, `SIGN_ERROR_LMS`, `RLS_LSTD`, or `ALL`.
- **`<ROWS>`, `<COLS>`:** Dimensions for the grid (e.g., `16`, `8`).
- **`<NUM_EPISODES>`:** Number of training episodes.
- **`<STRATEGY>`:** Feature encoding strategy (e.g., `one_hot_state_action`).

Example:

Run LMS on a 10x10 grid for 1000 episodes:

PowerShell

```
python .\main_experiment.py --algo LMS --grid_rows 10 --grid_cols 10 --episodes 1000
```

After running experiments, you can generate/update all plots and the CSV summary by running:

PowerShell

```
python .\plotting.py
```

## 7. Configuration

Global parameters and default settings can be modified in `config.py`. This includes:

- Grid dimensions and obstacle layouts (for 16x16 and 8x8).
- Reward values.
- RL parameters (gamma, epsilon schedule).
- Adaptive filter learning rates and specific parameters (e.g., RLS forgetting factor).
- Default number of episodes and steps per episode for training.
- Plotting colors and logging preferences.

It's generally recommended to use command-line arguments in `main_experiment.py` or modify `run_all_experiments.ps1` for specific experimental runs to preserve the default configurations in `config.py`.

## 8. Outputs

The project generates the following outputs:

- **Numerical Results (`results/` directory):**
    - `.npy` files for each experiment run, containing detailed data such as episode rewards, steps, paths, V-functions, timings, and a snapshot of the configuration used.
    - `experiment_summary.csv`: A CSV file summarizing key metrics and configurations from all conducted experiments.
- **Plots (`plots/` directory):**
    - Learning curves (cumulative reward vs. episode) for each individual run and combined comparisons.
    - Grid maps visualizing the environment, obstacles, start/goal positions, and the optimal vs. learned paths for each run.
    - Heatmaps of the learned state-value function (V(s)) for each run.
    - Bar charts comparing summary statistics across different algorithms and settings (e.g., training duration, final average reward, path length ratio).
