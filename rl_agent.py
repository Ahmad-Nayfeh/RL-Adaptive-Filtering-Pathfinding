# rl_agent.py (v2)

import numpy as np
import random
import config
from feature_encoder import FeatureEncoder
import adaptive_filters

class RLAgent:
    def __init__(self, feature_encoder_instance: FeatureEncoder, num_actions,
                 adaptive_filter_type='LMS', agent_params=None, filter_params=None):
        self.feature_encoder = feature_encoder_instance
        self.num_actions = num_actions
        self.adaptive_filter_type = adaptive_filter_type.upper()

        if agent_params is None: agent_params = {}
        self.gamma = agent_params.get('gamma', config.GAMMA)

        if filter_params is None: filter_params = {}
        self.feature_vector_size = self.feature_encoder.get_feature_vector_size()
        
        if self.adaptive_filter_type == 'RLS_LSTD':
            self.rls_updater = adaptive_filters.RLSUpdater(
                num_features=self.feature_vector_size,
                forgetting_factor=filter_params.get('rls_forgetting_factor', config.RLS_FORGETTING_FACTOR),
                initial_P_diag_value=filter_params.get('rls_init_p_diag_value', config.RLS_INIT_P_DIAG_VALUE)
            )
            self.weights = self.rls_updater.get_weights()
        else:
            self.weights = np.zeros(self.feature_vector_size)
            self.rls_updater = None

        self.lms_learning_rate = filter_params.get('lms_learning_rate', config.LMS_LEARNING_RATE)
        self.nlms_learning_rate = filter_params.get('nlms_learning_rate', config.NLMS_LEARNING_RATE)
        self.nlms_epsilon_norm = filter_params.get('nlms_epsilon_norm', config.NLMS_EPSILON_NORM)
        self.sign_error_lms_learning_rate = filter_params.get('sign_error_lms_learning_rate', config.SIGN_ERROR_LMS_LEARNING_RATE)

        if config.VERBOSE_LOGGING:
            print(f"RL Agent initialized with {self.adaptive_filter_type} filter.")
            print(f"Feature vector size: {self.feature_vector_size}, Gamma: {self.gamma}")
            if self.rls_updater:
                print(f"RLS Updater params: lambda={self.rls_updater.forgetting_factor}, P_init={self.rls_updater.initial_P_diag_value}")
            # else:
            #     print(f"LMS LR: {self.lms_learning_rate}, NLMS LR: {self.nlms_learning_rate}, SignErr LR: {self.sign_error_lms_learning_rate}")


    def _get_q_value(self, state, action):
        features = self.feature_encoder.get_features(state, action)
        if features.shape[0] != self.weights.shape[0]:
            raise ValueError(f"Feature vector size {features.shape[0]} != weights size {self.weights.shape[0]}.")
        return np.dot(self.weights, features)

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(config.ACTIONS)
        else:
            q_values = [self._get_q_value(state, action) for action in config.ACTIONS]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        q_current = self._get_q_value(state, action)
        if done:
            td_target = reward
        else:
            q_next_max = np.max([self._get_q_value(next_state, next_action) for next_action in config.ACTIONS])
            td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current
        features_sa = self.feature_encoder.get_features(state, action)

        if self.adaptive_filter_type == 'LMS':
            self.weights = adaptive_filters.lms_update(self.weights, td_error, features_sa, self.lms_learning_rate)
        elif self.adaptive_filter_type == 'NLMS':
            self.weights = adaptive_filters.nlms_update(self.weights, td_error, features_sa, self.nlms_learning_rate, self.nlms_epsilon_norm)
        elif self.adaptive_filter_type == 'SIGN_ERROR_LMS':
            self.weights = adaptive_filters.sign_error_lms_update(self.weights, td_error, features_sa, self.sign_error_lms_learning_rate)
        elif self.adaptive_filter_type == 'RLS_LSTD':
            if self.rls_updater is None: raise ValueError("RLSUpdater not initialized.")
            self.rls_updater.update(td_error, features_sa)
            self.weights = self.rls_updater.get_weights()
        else:
            raise ValueError(f"Unsupported adaptive filter type: {self.adaptive_filter_type}")

    def get_greedy_path(self, env_instance, max_steps=None):
        """
        Traces the path taken by the agent using a purely greedy policy (epsilon=0).
        Args:
            env_instance (GridEnvironment): An instance of the environment (used for its reset and step).
            max_steps (int, optional): Max steps for tracing. Defaults to config.MAX_STEPS_PER_EPISODE.
        Returns:
            list of tuples: The path as a list of (row, col) coordinates.
                            Returns None if goal not reached or error.
        """
        if max_steps is None:
            max_steps = config.MAX_STEPS_PER_EPISODE * 2 # Allow more steps for greedy path
        
        current_env_state = env_instance.reset() # Use a fresh reset of the passed env instance
        path = [current_env_state]
        
        for _ in range(max_steps):
            action = self.choose_action(current_env_state, epsilon=0.0) # Greedy action
            next_env_state, _, done, _ = env_instance.step(action)
            path.append(next_env_state)
            current_env_state = next_env_state
            if done:
                # Check if done was due to reaching goal
                if current_env_state == env_instance.goal_pos:
                    return path
                else: # Hit obstacle or other terminal state not goal
                    return path # Return path up to that point
        return path # Goal not reached within max_steps

    def get_value_function_grid(self, grid_rows, grid_cols, obstacles_set):
        """
        Calculates V(s) = max_a Q(s,a) for all non-obstacle states in the grid.
        Args:
            grid_rows (int): Number of rows in the grid.
            grid_cols (int): Number of columns in the grid.
            obstacles_set (set): Set of (r,c) obstacle coordinates.
        Returns:
            np.ndarray: A 2D array (grid_rows x grid_cols) representing V(s).
                        Obstacle cells can be NaN or a very small number.
        """
        v_grid = np.full((grid_rows, grid_cols), np.nan) # Initialize with NaN
        for r in range(grid_rows):
            for c in range(grid_cols):
                state = (r, c)
                if state not in obstacles_set:
                    q_values = [self._get_q_value(state, action) for action in config.ACTIONS]
                    v_grid[r, c] = np.max(q_values)
        return v_grid

# Example usage
if __name__ == "__main__":
    class MockFeatureEncoder:
        def __init__(self, vector_size, strategy_name="mock_phi_sa"):
            self._size = vector_size; self.strategy = strategy_name
        def get_features(self, state, action=None): return np.random.rand(self._size)
        def get_feature_vector_size(self): return self._size

    mock_encoder = MockFeatureEncoder(vector_size=10)
    agent = RLAgent(mock_encoder, config.NUM_ACTIONS, adaptive_filter_type='LMS')
    print("RLAgent with MockEncoder instantiated for basic checks.")
    # Further testing of get_greedy_path and get_value_function_grid
    # would require a mock environment.
