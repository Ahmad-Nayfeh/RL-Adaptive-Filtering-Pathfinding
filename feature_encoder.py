# feature_encoder.py (With xy_plus_action implemented)

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
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_actions = num_actions

        if strategy is None:
            self.strategy = config.FEATURE_ENCODING_STRATEGY
        else:
            self.strategy = strategy

        self._feature_vector_size = self._calculate_feature_vector_size()

        if config.VERBOSE_LOGGING:
            print(f"Feature Encoder initialized with strategy: '{self.strategy}'")
            print(f"Feature vector size: {self._feature_vector_size}")

    def _calculate_feature_vector_size(self):
        """
        Calculates the size of the feature vector based on the chosen strategy.
        """
        if self.strategy == 'simple_coords':
            return 2
        elif self.strategy == 'one_hot_state':
            return self.grid_rows * self.grid_cols
        elif self.strategy == 'one_hot_state_action':
            return (self.grid_rows * self.grid_cols) * self.num_actions
        elif self.strategy == 'xy_plus_action': # As suggested by ChatGPT
            return 2 + self.num_actions # [norm_row, norm_col, one_hot_action...]
        else:
            raise ValueError(f"Unsupported feature encoding strategy: {self.strategy}")

    def get_features(self, state, action=None):
        """
        Generates the feature vector for a given state (and optionally action).
        """
        row, col = state
        # Initialize feature_vector AFTER _feature_vector_size is confirmed for the current strategy
        # This ensures if strategy is changed after init (not typical but possible), size is correct.
        # However, _feature_vector_size is set in __init__ based on self.strategy, so it should be fine.
        feature_vector = np.zeros(self._feature_vector_size)


        if self.strategy == 'simple_coords':
            feature_vector[0] = row
            feature_vector[1] = col
        elif self.strategy == 'one_hot_state':
            index = row * self.grid_cols + col
            if 0 <= index < feature_vector.shape[0]:
                feature_vector[index] = 1.0
            else:
                if config.VERBOSE_LOGGING:
                    print(f"Warning: Index {index} out of bounds for one_hot_state encoding.")
        elif self.strategy == 'one_hot_state_action':
            if action is None:
                raise ValueError("Action must be provided for 'one_hot_state_action' strategy.")
            state_index = row * self.grid_cols + col
            action_index = action
            combined_index = state_index * self.num_actions + action_index
            if 0 <= combined_index < feature_vector.shape[0]:
                feature_vector[combined_index] = 1.0
            else:
                if config.VERBOSE_LOGGING:
                    print(f"Warning: Index {combined_index} out of bounds for one_hot_state_action encoding.")
        elif self.strategy == 'xy_plus_action': # Implementation for ChatGPT's suggestion
            if action is None:
                raise ValueError("Action must be provided for 'xy_plus_action' strategy.")
            
            # Normalized coordinates
            # Denominator should be (max_value) to scale to [0,1]. Max index is size-1.
            # If grid_rows/cols is 1, (size-1) would be 0. Handle this.
            norm_row = row / (self.grid_rows - 1.0) if self.grid_rows > 1 else 0.0
            norm_col = col / (self.grid_cols - 1.0) if self.grid_cols > 1 else 0.0
            
            feature_vector[0] = norm_row
            feature_vector[1] = norm_col
            
            # One-hot encode the action, starting from index 2
            if 0 <= action < self.num_actions:
                feature_vector[2 + action] = 1.0
            else:
                # This case should ideally not be reached if agent action selection is correct
                if config.VERBOSE_LOGGING:
                    print(f"Warning: Invalid action index {action} for xy_plus_action encoding.")
        else:
            raise ValueError(f"Unsupported feature encoding strategy in get_features: {self.strategy}")

        return feature_vector

    def get_feature_vector_size(self):
        """
        Returns the size (dimensionality) of the feature vector.
        """
        return self._feature_vector_size

# Example usage (for testing this file directly)
if __name__ == "__main__":
    rows = config.GRID_ROWS
    cols = config.GRID_COLS
    n_actions = config.NUM_ACTIONS

    print(f"\n--- Testing 'xy_plus_action' strategy (NEW) ---")
    encoder_xy_action = FeatureEncoder(rows, cols, n_actions, strategy='xy_plus_action')
    test_state_xy = (3, 5) # Example state (row=3, col=5)
    test_action_xy = config.ACTION_UP # Example action (0)
    features_xy_action = encoder_xy_action.get_features(test_state_xy, action=test_action_xy)
    print(f"State: {test_state_xy}, Action: {test_action_xy}")
    print(f"Features ('xy_plus_action'): {features_xy_action}")
    print(f"Expected size: {2 + n_actions}, Actual size: {encoder_xy_action.get_feature_vector_size()}")
    # Expected: [3/15, 5/15, 1, 0, 0, 0] for 16x16 grid (0-indexed)
    # For (3,5) on 16x16, norm_row=3/15=0.2, norm_col=5/15=0.333. If action is UP (0), then [0.2, 0.333, 1, 0, 0, 0]

    print(f"\n--- Testing 'one_hot_state_action' strategy (for comparison) ---")
    encoder_one_hot_sa = FeatureEncoder(rows, cols, n_actions, strategy='one_hot_state_action')
    # test_state_sa = (0,1)
    # test_action_sa = 1
    # features_one_hot_sa = encoder_one_hot_sa.get_features(test_state_sa, action=test_action_sa)
    # print(f"State: {test_state_sa}, Action: {test_action_sa}")
    # print(f"Features ('one_hot_state_action') sample (index {0*cols*n_actions + 1*n_actions + 1} should be 1.0)")
    # print(f"Feature vector size: {encoder_one_hot_sa.get_feature_vector_size()}")
    print(f"Feature vector size for 'one_hot_state_action': {encoder_one_hot_sa.get_feature_vector_size()}")