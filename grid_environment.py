# grid_environment.py (v4 - with reward shaping)

import numpy as np
import collections
import config

class GridEnvironment:
    def __init__(self):
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS
        
        if self.grid_rows == 16 and self.grid_cols == 16:
            self.start_pos = config.START_POS
            self.goal_pos = config.GOAL_POS
            current_obstacle_list = config.OBSTACLES_16x16
            if config.VERBOSE_LOGGING: print("Using 16x16 obstacle set.")
        elif self.grid_rows == 8 and self.grid_cols == 8:
            self.start_pos = config.START_POS_8x8
            self.goal_pos = config.GOAL_POS_8x8
            current_obstacle_list = config.OBSTACLES_8x8
            if config.VERBOSE_LOGGING: print("Using 8x8 obstacle set.")
        else:
            self.start_pos = (0,0)
            self.goal_pos = (self.grid_rows - 1, self.grid_cols - 1)
            current_obstacle_list = []
            if config.VERBOSE_LOGGING: print(f"Warning: No specific obstacle set for {self.grid_rows}x{self.grid_cols}. Using empty.")

        self.obstacles = set()
        for r_obs, c_obs in current_obstacle_list:
            if 0 <= r_obs < self.grid_rows and 0 <= c_obs < self.grid_cols:
                if (r_obs, c_obs) != self.start_pos and (r_obs, c_obs) != self.goal_pos:
                    self.obstacles.add((r_obs, c_obs))
        
        # For other modules that might refer to config for dynamic values
        config.START_POS_DYNAMIC = self.start_pos 
        config.GOAL_POS_DYNAMIC = self.goal_pos
        config.OBSTACLES_DYNAMIC = list(self.obstacles)

        self.agent_pos = None
        self.action_space = config.ACTIONS
        self.num_actions = config.NUM_ACTIONS

        if config.VERBOSE_LOGGING:
            print(f"Grid Environment initialized: {self.grid_rows}x{self.grid_cols}")
            print(f"Effective Start: {self.start_pos}, Effective Goal: {self.goal_pos}")
            print(f"Reward Shaping Factor: {config.REWARD_SHAPING_FACTOR}")

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def _is_valid_position(self, pos):
        row, col = pos
        return 0 <= row < self.grid_rows and 0 <= col < self.grid_cols

    def _is_obstacle(self, pos):
        return pos in self.obstacles

    def step(self, action):
        if self.agent_pos is None:
            raise ValueError("Agent position is not initialized. Call reset() first.")

        current_row, current_col = self.agent_pos
        next_row, next_col = current_row, current_col

        if action == config.ACTION_UP: next_row -= 1
        elif action == config.ACTION_DOWN: next_row += 1
        elif action == config.ACTION_LEFT: next_col -= 1
        elif action == config.ACTION_RIGHT: next_col += 1
        else: raise ValueError(f"Invalid action: {action}.")

        next_pos = (next_row, next_col)
        reward = 0
        done = False
        info = {}

        if not self._is_valid_position(next_pos):
            reward = config.REWARD_INVALID_MOVE
            # Agent position does not change for invalid move
            next_state = self.agent_pos 
        elif self._is_obstacle(next_pos):
            reward = config.REWARD_OBSTACLE
            # Agent position does not change if it hits an obstacle
            next_state = self.agent_pos
        elif next_pos == self.goal_pos:
            reward = config.REWARD_GOAL
            self.agent_pos = next_pos
            next_state = self.agent_pos
            done = True
        else: # Regular valid step
            # Standard step penalty
            step_reward = config.REWARD_STEP
            
            # Reward Shaping: Penalty based on Manhattan distance to goal
            if config.REWARD_SHAPING_FACTOR != 0:
                manhattan_distance = abs(next_pos[0] - self.goal_pos[0]) + abs(next_pos[1] - self.goal_pos[1])
                shaping_penalty = config.REWARD_SHAPING_FACTOR * manhattan_distance
                reward = step_reward - shaping_penalty # Subtract penalty (factor is positive)
            else:
                reward = step_reward

            self.agent_pos = next_pos
            next_state = self.agent_pos
            
        return next_state, reward, done, info

    def render(self, mode='human', path_to_draw=None):
        if mode == 'human':
            grid_repr = np.full((self.grid_rows, self.grid_cols), '.', dtype=str)
            if self.start_pos: grid_repr[self.start_pos] = 'S'
            if self.goal_pos: grid_repr[self.goal_pos] = 'G'
            for obs_pos in self.obstacles: grid_repr[obs_pos] = 'X'
            if path_to_draw:
                for r_idx,c_idx in path_to_draw:
                    if (r_idx,c_idx) != self.start_pos and (r_idx,c_idx) != self.goal_pos and (r_idx,c_idx) not in self.obstacles :
                        grid_repr[r_idx,c_idx] = '*' # Path marker
            if self.agent_pos and grid_repr[self.agent_pos] == '.': # Don't overwrite S, G, X
                grid_repr[self.agent_pos] = 'A'
            
            for r_idx in range(self.grid_rows): print(" ".join(grid_repr[r_idx]))
            print("-" * (self.grid_cols * 2 -1))

    def get_optimal_path_bfs(self):
        if not self.start_pos or not self.goal_pos: return None
        queue = collections.deque([([self.start_pos], self.start_pos)])
        visited = {self.start_pos}
        while queue:
            current_path, (r, c) = queue.popleft()
            if (r, c) == self.goal_pos: return current_path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Up, Down, Left, Right
                nr, nc = r + dr, c + dc
                next_node = (nr, nc)
                if self._is_valid_position(next_node) and not self._is_obstacle(next_node) and next_node not in visited:
                    visited.add(next_node)
                    new_path = list(current_path) # Create a new list for the new path
                    new_path.append(next_node)
                    queue.append((new_path, next_node))
        return None

if __name__ == "__main__":
    print("--- Testing Grid Environment with Reward Shaping ---")
    # Test with 8x8
    config.GRID_ROWS = 8
    config.GRID_COLS = 8
    config.REWARD_SHAPING_FACTOR = 0.1 # Enable shaping for test
    env8 = GridEnvironment()
    state = env8.reset()
    print(f"Initial state: {state}")
    env8.render()

    # Test a few steps
    actions_to_test = [config.ACTION_RIGHT, config.ACTION_RIGHT, config.ACTION_DOWN]
    for action_idx, action in enumerate(actions_to_test):
        next_s, reward, done, _ = env8.step(action)
        print(f"Step {action_idx+1}: Action={action}, NextState={next_s}, Reward={reward:.2f}, Done={done}")
        env8.render()
        if done: break
    
    optimal_path = env8.get_optimal_path_bfs()
    if optimal_path:
        print(f"Optimal path for 8x8: {optimal_path}")
        print(f"Optimal path length: {len(optimal_path)-1}")
    else:
        print("No optimal path found for 8x8.")