# grid_environment.py (v3 - uses new obstacle configs)

import numpy as np
import collections # For deque in BFS
import config

class GridEnvironment:
    def __init__(self):
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS
        
        # Dynamically select obstacles and start/goal based on grid size
        if self.grid_rows == 16 and self.grid_cols == 16:
            self.start_pos = config.START_POS # Uses global START_POS
            self.goal_pos = config.GOAL_POS   # Uses global GOAL_POS
            current_obstacle_list = config.OBSTACLES_16x16
            if config.VERBOSE_LOGGING: print("Using 16x16 obstacle set.")
        elif self.grid_rows == 8 and self.grid_cols == 8:
            self.start_pos = config.START_POS_8x8
            self.goal_pos = config.GOAL_POS_8x8
            current_obstacle_list = config.OBSTACLES_8x8
            if config.VERBOSE_LOGGING: print("Using 8x8 obstacle set.")
        else: # Fallback for other sizes - use empty obstacle list or a generic one
            self.start_pos = (0,0)
            self.goal_pos = (self.grid_rows - 1, self.grid_cols - 1)
            current_obstacle_list = []
            if config.VERBOSE_LOGGING: print(f"Warning: No specific obstacle set for {self.grid_rows}x{self.grid_cols}. Using empty.")

        # Ensure obstacles are valid for the current grid dimensions and don't include start/goal
        self.obstacles = set()
        for r_obs, c_obs in current_obstacle_list:
            if 0 <= r_obs < self.grid_rows and 0 <= c_obs < self.grid_cols:
                if (r_obs, c_obs) != self.start_pos and (r_obs, c_obs) != self.goal_pos:
                    self.obstacles.add((r_obs, c_obs))
        
        # Update config's global START_POS and GOAL_POS to reflect what this instance is using
        # This helps if other modules (like plotting) refer to config.START_POS directly for this env.
        # However, it's better if plotting gets S/G from the saved results' config_snapshot.
        # For now, this ensures the env instance's S/G are definitive for its operations.
        config.START_POS_DYNAMIC = self.start_pos # Store the dynamically chosen start/goal
        config.GOAL_POS_DYNAMIC = self.goal_pos
        config.OBSTACLES_DYNAMIC = list(self.obstacles)


        self.agent_pos = None
        self.action_space = config.ACTIONS
        self.num_actions = config.NUM_ACTIONS

        if config.VERBOSE_LOGGING:
            print(f"Grid Environment initialized: {self.grid_rows}x{self.grid_cols}")
            print(f"Effective Start: {self.start_pos}, Effective Goal: {self.goal_pos}")

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
            next_state = self.agent_pos
        elif self._is_obstacle(next_pos):
            reward = config.REWARD_OBSTACLE
            next_state = self.agent_pos
        elif next_pos == self.goal_pos:
            reward = config.REWARD_GOAL
            self.agent_pos = next_pos
            next_state = self.agent_pos
            done = True
        else:
            reward = config.REWARD_STEP
            self.agent_pos = next_pos
            next_state = self.agent_pos
        return next_state, reward, done, info

    def render(self, mode='human', path_to_draw=None, path_color='blue'): # Unchanged from v2
        if mode == 'human':
            grid_repr = np.full((self.grid_rows, self.grid_cols), '.', dtype=str)
            if self.start_pos: grid_repr[self.start_pos] = 'S'
            if self.goal_pos: grid_repr[self.goal_pos] = 'G'
            for obs_pos in self.obstacles: grid_repr[obs_pos] = 'X'
            if path_to_draw:
                for r,c in path_to_draw:
                    if (r,c) != self.start_pos and (r,c) != self.goal_pos: grid_repr[r,c] = '*'
            if self.agent_pos: grid_repr[self.agent_pos] = 'A'
            for r_idx in range(self.grid_rows): print(" ".join(grid_repr[r_idx]))
            print("-" * (self.grid_cols * 2 -1))

    def get_optimal_path_bfs(self): # Unchanged from v2
        if not self.start_pos or not self.goal_pos: return None
        queue = collections.deque([([self.start_pos], self.start_pos)])
        visited = {self.start_pos}
        while queue:
            current_path, (r, c) = queue.popleft()
            if (r, c) == self.goal_pos: return current_path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc; next_node = (nr, nc)
                if self._is_valid_position(next_node) and not self._is_obstacle(next_node) and next_node not in visited:
                    visited.add(next_node); new_path = current_path + [next_node]; queue.append((new_path, next_node))
        return None

if __name__ == "__main__":
    # Test with 16x16 default from config
    print("--- Testing 16x16 Environment (Harder Obstacles) ---")
    config.GRID_ROWS = 16 # Ensure config matches test
    config.GRID_COLS = 16
    env16 = GridEnvironment()
    env16.reset()
    env16.render()
    optimal_path16 = env16.get_optimal_path_bfs()
    if optimal_path16:
        print(f"16x16 Optimal path length: {len(optimal_path16) -1 } steps")
        # env16.render(path_to_draw=optimal_path16) # Can be very long
    else:
        print("16x16 No optimal path found (check obstacles vs start/goal).")

    print("\n--- Testing 8x8 Environment (Harder Obstacles) ---")
    config.GRID_ROWS = 8 # Temporarily change config for testing 8x8
    config.GRID_COLS = 8
    env8 = GridEnvironment()
    env8.reset()
    env8.render()
    optimal_path8 = env8.get_optimal_path_bfs()
    if optimal_path8:
        print(f"8x8 Optimal path length: {len(optimal_path8) -1 } steps")
        # env8.render(path_to_draw=optimal_path8)
    else:
        print("8x8 No optimal path found (check obstacles vs start/goal).")
