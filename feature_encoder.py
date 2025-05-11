# feature_encoder.py

import numpy as np
import config # Import configurations

class FeatureEncoder:
    """
    Encodes the state (and optionally action) from the environment into a feature vector.
    This vector is used by the linear function approximator in the RL agent.
    """

    def __init__(self, grid_rows, grid_cols, num_actions, strategy=None):
        """
        Initializes the feature encoder.

        Args:
            grid_rows (int): Number of rows in the grid.
            grid_cols (int): Number of columns in the grid.
            num_actions (int): Number of possible actions the agent can take.
            strategy (str, optional): The encoding strategy to use.
                                      Defaults to config.FEATURE_ENCODING_STRATEGY.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_actions = num_actions # Kept for potential future strategies that include actions

        if strategy is None:
            self.strategy = config.FEATURE_ENCODING_STRATEGY
        else:
            self.strategy = strategy

        self._feature_vector_size = self._calculate_feature_vector_size()

        print(f"Feature Encoder initialized with strategy: '{self.strategy}'")
        print(f"Feature vector size: {self._feature_vector_size}")

    def _calculate_feature_vector_size(self):
        """
        Calculates the size of the feature vector based on the chosen strategy.
        """
        if self.strategy == 'simple_coords':
            # Features: [row, col] - represents the agent's position
            # We could also add a bias term: [row, col, 1]
            return 2 # For (row, col)
            # return 3 # For (row, col, bias_term)
        elif self.strategy == 'one_hot_state':
            # One-hot encoding for each possible state (cell)
            return self.grid_rows * self.grid_cols
        elif self.strategy == 'one_hot_state_action':
            # One-hot encoding for each state, concatenated for each action,
            # or a single large vector where only one (state,action) pair is active.
            # This depends on how Q(s,a) is structured.
            # For Q(s,a) = w^T phi(s,a), phi(s,a) is often state features repeated for each action,
            # with zeros for other actions, or features specific to the (s,a) pair.
            # A simpler Q(s,a) approach: learn separate weight vectors w_a for each action,
            # so Q(s,a) = w_a^T phi(s). In this case, phi(s) is just state features.
            # For now, let's assume phi(s,a) will be specific.
            # If phi(s,a) means features of s concatenated for each action, and only one action's features are non-zero:
            # Example: state_features = [r,c]. phi(s,a_0) = [r,c,0,0,0,0], phi(s,a_1) = [0,0,r,c,0,0] ...
            # This would make the feature vector size = (state_feature_size * num_actions)
            # For simplicity in this example, let's assume one_hot_state_action means
            # a unique feature for each (state, action) pair.
            return (self.grid_rows * self.grid_cols) * self.num_actions
        # Add other strategies like 'tile_coding' here later
        else:
            raise ValueError(f"Unsupported feature encoding strategy: {self.strategy}")

    def get_features(self, state, action=None):
        """
        Generates the feature vector for a given state (and optionally action).

        Args:
            state (tuple): The agent's current position (row, col).
            action (int, optional): The action taken or being considered.
                                    Required for strategies like 'one_hot_state_action'.

        Returns:
            numpy.ndarray: The feature vector.
        """
        row, col = state
        feature_vector = np.zeros(self._feature_vector_size)

        if self.strategy == 'simple_coords':
            # Simple (row, col) as features.
            # Normalization might be useful here if coordinates vary widely, but for a fixed grid, it's often okay.
            feature_vector[0] = row
            feature_vector[1] = col
            # if self._feature_vector_size == 3: # If using a bias term
            #     feature_vector[2] = 1.0 # Bias term
        elif self.strategy == 'one_hot_state':
            index = row * self.grid_cols + col
            if 0 <= index < feature_vector.shape[0]:
                feature_vector[index] = 1.0
            else:
                print(f"Warning: Index {index} out of bounds for one_hot_state encoding.")
        elif self.strategy == 'one_hot_state_action':
            if action is None:
                raise ValueError("Action must be provided for 'one_hot_state_action' strategy.")
            state_index = row * self.grid_cols + col
            action_index = action
            # Unique index for each (state, action) pair
            combined_index = state_index * self.num_actions + action_index
            if 0 <= combined_index < feature_vector.shape[0]:
                feature_vector[combined_index] = 1.0
            else:
                print(f"Warning: Index {combined_index} out of bounds for one_hot_state_action encoding.")
        else:
            # This case should have been caught in init, but as a safeguard:
            raise ValueError(f"Unsupported feature encoding strategy: {self.strategy}")

        return feature_vector

    def get_feature_vector_size(self):
        """
        Returns the size (dimensionality) of the feature vector.

        Returns:
            int: The size of the feature vector.
        """
        return self._feature_vector_size

# Example usage (for testing this file directly)
if __name__ == "__main__":
    # Use config for grid dimensions and num_actions
    rows = config.GRID_ROWS
    cols = config.GRID_COLS
    n_actions = config.NUM_ACTIONS

    print(f"\n--- Testing 'simple_coords' strategy ---")
    encoder_simple = FeatureEncoder(rows, cols, n_actions, strategy='simple_coords')
    test_state = (3, 5) # Example state
    features_simple = encoder_simple.get_features(test_state)
    print(f"State: {test_state}, Features ('simple_coords'): {features_simple}")
    print(f"Expected size: 2, Actual size: {encoder_simple.get_feature_vector_size()}")

    # print(f"\n--- Testing 'one_hot_state' strategy ---")
    # # For a 16x16 grid, one_hot_state will be 256 features.
    # # To make example output readable, let's test with a smaller hypothetical grid for printing.
    # small_rows, small_cols = 3, 3
    # encoder_one_hot = FeatureEncoder(small_rows, small_cols, n_actions, strategy='one_hot_state')
    # test_state_small = (1, 1) # (row=1, col=1)
    # # Expected index: 1 * 3 (cols) + 1 = 4
    # features_one_hot = encoder_one_hot.get_features(test_state_small)
    # print(f"State: {test_state_small} on {small_rows}x{small_cols} grid")
    # print(f"Features ('one_hot_state', index 4 should be 1.0): {features_one_hot}")
    # print(f"Expected size: {small_rows*small_cols}, Actual size: {encoder_one_hot.get_feature_vector_size()}")

    # print(f"\n--- Testing 'one_hot_state_action' strategy ---")
    # # Test with a small grid and few actions for readable output
    # small_rows_sa, small_cols_sa = 2, 2
    # small_n_actions = 2
    # encoder_one_hot_sa = FeatureEncoder(small_rows_sa, small_cols_sa, small_n_actions, strategy='one_hot_state_action')
    # test_state_sa = (0,1) # state_index = 0 * 2 + 1 = 1
    # test_action_sa = 1    # action_index = 1
    # # Expected combined_index = state_index * num_actions + action_index = 1 * 2 + 1 = 3
    # features_one_hot_sa = encoder_one_hot_sa.get_features(test_state_sa, action=test_action_sa)
    # print(f"State: {test_state_sa}, Action: {test_action_sa} on {small_rows_sa}x{small_cols_sa} grid, {small_n_actions} actions")
    # print(f"Features ('one_hot_state_action', index 3 should be 1.0): {features_one_hot_sa}")
    # print(f"Expected size: {(small_rows_sa*small_cols_sa)*small_n_actions}, Actual size: {encoder_one_hot_sa.get_feature_vector_size()}")

    # Re-enable for full grid if desired, but output will be large for one-hot
    print(f"\n--- Testing 'one_hot_state' strategy (full {rows}x{cols} grid) ---")
    encoder_one_hot_full = FeatureEncoder(rows, cols, n_actions, strategy='one_hot_state')
    test_state_full = (0,0)
    features_one_hot_full = encoder_one_hot_full.get_features(test_state_full)
    print(f"State: {test_state_full}, Features ('one_hot_state') for {rows}x{cols} grid (first element should be 1.0):")
    # print(features_one_hot_full) # This will be a large array
    print(f"Feature vector (sum should be 1.0): sum = {np.sum(features_one_hot_full)}")
    print(f"Feature vector (shape): {features_one_hot_full.shape}")


