# adaptive_filters.py

import numpy as np
import config # Import configurations

# --- LMS Family ---

def lms_update(weights, error, features, learning_rate=None):
    """
    Performs a weight update using the standard Least Mean Squares (LMS) algorithm.

    Update rule: w_new = w_old + learning_rate * error * features

    Args:
        weights (np.ndarray): The current weight vector.
        error (float): The error signal (e.g., TD error).
        features (np.ndarray): The feature vector corresponding to the current state/action.
        learning_rate (float, optional): The learning rate (mu).
                                         Defaults to config.LMS_LEARNING_RATE.

    Returns:
        np.ndarray: The updated weight vector.
    """
    if learning_rate is None:
        learning_rate = config.LMS_LEARNING_RATE
    
    if not isinstance(weights, np.ndarray) or not isinstance(features, np.ndarray):
        raise TypeError("Weights and features must be numpy arrays.")
    if weights.shape != features.shape:
        # Allow for weights (N,) and features (N,) or weights (N,1) and features (N,1) or (1,N)
        if weights.ndim == 1 and features.ndim == 1 and weights.shape[0] == features.shape[0]:
            pass # This is fine
        elif weights.size == features.size: # If they can be reshaped to match
             features = features.reshape(weights.shape) # Attempt to reshape features to match weights
        else:
            raise ValueError(f"Weights shape {weights.shape} and features shape {features.shape} are incompatible.")

    delta_weights = learning_rate * error * features
    updated_weights = weights + delta_weights
    return updated_weights

def nlms_update(weights, error, features, learning_rate=None, epsilon_norm=None):
    """
    Performs a weight update using the Normalized Least Mean Squares (NLMS) algorithm.

    Update rule: w_new = w_old + (learning_rate / (epsilon_norm + ||features||^2)) * error * features

    Args:
        weights (np.ndarray): The current weight vector.
        error (float): The error signal (e.g., TD error).
        features (np.ndarray): The feature vector.
        learning_rate (float, optional): The learning rate (mu).
                                         Defaults to config.NLMS_LEARNING_RATE.
        epsilon_norm (float, optional): A small constant added to the denominator
                                        for numerical stability and to prevent division by zero.
                                        Defaults to config.NLMS_EPSILON_NORM.

    Returns:
        np.ndarray: The updated weight vector.
    """
    if learning_rate is None:
        learning_rate = config.NLMS_LEARNING_RATE
    if epsilon_norm is None:
        epsilon_norm = config.NLMS_EPSILON_NORM

    if not isinstance(weights, np.ndarray) or not isinstance(features, np.ndarray):
        raise TypeError("Weights and features must be numpy arrays.")
    if weights.shape != features.shape:
        if weights.ndim == 1 and features.ndim == 1 and weights.shape[0] == features.shape[0]:
            pass
        elif weights.size == features.size:
             features = features.reshape(weights.shape)
        else:
            raise ValueError(f"Weights shape {weights.shape} and features shape {features.shape} are incompatible.")

    norm_squared_features = np.sum(features**2) # or np.dot(features, features)
    
    # Ensure denominator is not too small or zero
    denominator = epsilon_norm + norm_squared_features
    if denominator < 1e-9: # Threshold to prevent very large updates if features are near zero
        effective_learning_rate = learning_rate 
    else:
        effective_learning_rate = learning_rate / denominator
        
    delta_weights = effective_learning_rate * error * features
    updated_weights = weights + delta_weights
    return updated_weights

def sign_error_lms_update(weights, error, features, learning_rate=None):
    """
    Performs a weight update using the Sign-Error LMS algorithm.

    Update rule: w_new = w_old + learning_rate * sign(error) * features

    Args:
        weights (np.ndarray): The current weight vector.
        error (float): The error signal (e.g., TD error).
        features (np.ndarray): The feature vector.
        learning_rate (float, optional): The learning rate (mu).
                                         Defaults to config.SIGN_ERROR_LMS_LEARNING_RATE.

    Returns:
        np.ndarray: The updated weight vector.
    """
    if learning_rate is None:
        learning_rate = config.SIGN_ERROR_LMS_LEARNING_RATE
    
    if not isinstance(weights, np.ndarray) or not isinstance(features, np.ndarray):
        raise TypeError("Weights and features must be numpy arrays.")
    if weights.shape != features.shape:
        if weights.ndim == 1 and features.ndim == 1 and weights.shape[0] == features.shape[0]:
            pass
        elif weights.size == features.size:
             features = features.reshape(weights.shape)
        else:
            raise ValueError(f"Weights shape {weights.shape} and features shape {features.shape} are incompatible.")

    delta_weights = learning_rate * np.sign(error) * features
    updated_weights = weights + delta_weights
    return updated_weights

# --- RLS Family ---

class RLSUpdater:
    """
    Implements the Recursive Least Squares (RLS) algorithm for updating weights.
    This is often used in the context of Least Squares Temporal Difference (LSTD).
    """
    def __init__(self, num_features, forgetting_factor=None, initial_P_diag_value=None):
        """
        Initializes the RLS updater.

        Args:
            num_features (int): The dimensionality of the feature vector (and weight vector).
            forgetting_factor (float, optional): Lambda (λ), controls how much past data is forgotten.
                                                Defaults to config.RLS_FORGETTING_FACTOR.
            initial_P_diag_value (float, optional): Value to initialize the diagonal of the
                                                   P matrix (inverse of the correlation matrix estimate).
                                                   A large value means low confidence in initial weights.
                                                   Defaults to config.RLS_INIT_P_DIAG_VALUE.
        """
        self.num_features = num_features
        
        if forgetting_factor is None:
            self.forgetting_factor = config.RLS_FORGETTING_FACTOR
        else:
            self.forgetting_factor = forgetting_factor
            
        if initial_P_diag_value is None:
            self.initial_P_diag_value = config.RLS_INIT_P_DIAG_VALUE
        else:
            self.initial_P_diag_value = initial_P_diag_value

        # Initialize weights w to zeros
        self.weights = np.zeros(num_features)
        
        # Initialize P matrix (inverse of feature correlation matrix estimate)
        # P = (1/delta) * I, where delta is a small positive constant.
        # Or P = large_value * I
        self.P_matrix = np.eye(num_features) * self.initial_P_diag_value
        
        print(f"RLS Updater initialized: {num_features} features, lambda={self.forgetting_factor}")

    def update(self, error, features):
        """
        Performs one RLS update step.

        Args:
            error (float): The error signal (e.g., TD error, which is d_k - y_k = d_k - features_k^T * w_{k-1}).
            features (np.ndarray): The feature vector (phi_k). Should be a 1D array of size num_features.

        Returns:
            np.ndarray: The updated weight vector.
        """
        if not isinstance(features, np.ndarray) or features.ndim != 1 or features.shape[0] != self.num_features:
            raise ValueError(f"Features must be a 1D numpy array of size {self.num_features}")

        # Ensure features is a column vector for matrix operations if needed,
        # but numpy handles 1D arrays well with np.dot and broadcasting.
        # For clarity in matrix equations, phi is often (num_features, 1)
        phi_k = features # Using 1D array directly

        # Calculate gain vector k_k:
        # k_k = (P_{k-1} * phi_k) / (lambda + phi_k^T * P_{k-1} * phi_k)
        P_phi = np.dot(self.P_matrix, phi_k) # P_{k-1} * phi_k
        phi_P_phi = np.dot(phi_k.T, P_phi)    # phi_k^T * P_{k-1} * phi_k
        
        denominator_gain = self.forgetting_factor + phi_P_phi
        if np.abs(denominator_gain) < 1e-9: # Avoid division by zero
            # This case might indicate issues with features or P matrix (e.g. P became zero)
            # Or if features are consistently zero.
            # One option is to skip update or reinitialize P, but that's complex.
            # For now, if denominator is zero, gain vector elements will be large/inf.
            # Let's make gain zero to prevent weight explosion, though this isn't ideal RLS.
            gain_vector_k = np.zeros_like(phi_k)
            if config.VERBOSE_LOGGING:
                print("Warning: RLS gain vector denominator is near zero. Gain set to zero.")
        else:
            gain_vector_k = P_phi / denominator_gain

        # Update weights w_k:
        # w_k = w_{k-1} + k_k * error_k
        # Note: error_k is (desired_signal_k - features_k^T * w_{k-1})
        # The 'error' passed to this function is assumed to be this prediction error.
        self.weights = self.weights + gain_vector_k * error

        # Update P matrix P_k:
        # P_k = (1/lambda) * (P_{k-1} - k_k * phi_k^T * P_{k-1})
        # P_k = (1/lambda) * (I - k_k * phi_k^T) * P_{k-1}
        k_phi_T = np.outer(gain_vector_k, phi_k.T) # k_k * phi_k^T (results in a matrix)
        identity_matrix = np.eye(self.num_features)
        self.P_matrix = (1.0 / self.forgetting_factor) * np.dot(identity_matrix - k_phi_T, self.P_matrix)

        return self.weights

    def get_weights(self):
        """Returns the current weight vector."""
        return self.weights

# Example usage (for testing this file directly)
if __name__ == "__main__":
    num_features_test = 3
    test_weights = np.array([0.1, 0.2, 0.3])
    test_features = np.array([1.0, 2.0, 0.5])
    test_error = 0.5 # Example error (d - y)

    print("--- Testing LMS ---")
    updated_w_lms = lms_update(np.copy(test_weights), test_error, test_features)
    print(f"Initial weights: {test_weights}")
    print(f"Features: {test_features}, Error: {test_error}")
    print(f"LMS Learning Rate: {config.LMS_LEARNING_RATE}")
    print(f"Updated LMS weights: {updated_w_lms}")

    print("\n--- Testing NLMS ---")
    updated_w_nlms = nlms_update(np.copy(test_weights), test_error, test_features)
    print(f"Initial weights: {test_weights}")
    print(f"Features: {test_features}, Error: {test_error}")
    print(f"NLMS Learning Rate: {config.NLMS_LEARNING_RATE}, Epsilon: {config.NLMS_EPSILON_NORM}")
    print(f"Updated NLMS weights: {updated_w_nlms}")

    print("\n--- Testing Sign-Error LMS ---")
    updated_w_se_lms = sign_error_lms_update(np.copy(test_weights), test_error, test_features)
    print(f"Initial weights: {test_weights}")
    print(f"Features: {test_features}, Error: {test_error}")
    print(f"Sign-Error LMS Learning Rate: {config.SIGN_ERROR_LMS_LEARNING_RATE}")
    print(f"Updated Sign-Error LMS weights: {updated_w_se_lms}")
    
    # Test with negative error for sign-error
    test_error_neg = -0.5
    updated_w_se_lms_neg = sign_error_lms_update(np.copy(test_weights), test_error_neg, test_features)
    print(f"Features: {test_features}, Error (Negative): {test_error_neg}")
    print(f"Updated Sign-Error LMS weights (neg error): {updated_w_se_lms_neg}")


    print("\n--- Testing RLS Updater ---")
    rls_updater = RLSUpdater(num_features=num_features_test)
    # RLS weights are initialized to zeros by the class, so we use those.
    initial_rls_weights = np.copy(rls_updater.get_weights()) 
    print(f"Initial RLS weights: {initial_rls_weights}")
    print(f"Initial P matrix (sample):\n{rls_updater.P_matrix[:2,:2]}...") # Print a sample

    # Simulate a few updates for RLS
    # For RLS, the error is d_k - phi_k^T * w_{k-1}
    # Let's assume a desired signal d_k for testing RLS
    desired_signals = [1.0, 0.5, 1.2]
    feature_stream = [np.array([1.0, 0.5, 0.2]), 
                      np.array([0.3, 1.0, 0.8]), 
                      np.array([0.7, 0.2, 1.0])]

    for i in range(len(desired_signals)):
        current_features = feature_stream[i]
        d_k = desired_signals[i]
        
        # Calculate prediction y_k = phi_k^T * w_{k-1}
        y_k = np.dot(current_features, rls_updater.get_weights())
        rls_error = d_k - y_k # This is the error RLS uses internally
        
        print(f"\nRLS Update {i+1}:")
        print(f"  Features (phi_k): {current_features}")
        print(f"  Desired (d_k): {d_k}")
        print(f"  Prediction (y_k): {y_k:.4f}")
        print(f"  Error (e_k = d_k - y_k): {rls_error:.4f}")
        
        updated_w_rls = rls_updater.update(rls_error, current_features)
        print(f"  Updated RLS weights: {updated_w_rls}")
        # print(f"  Updated P matrix (sample):\n{rls_updater.P_matrix[:2,:2]}...")
