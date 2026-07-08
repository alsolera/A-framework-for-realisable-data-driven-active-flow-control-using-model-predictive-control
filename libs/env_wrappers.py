import gymnasium as gym
import numpy as np
from collections import deque # Use deque for efficient pops from left

class DelayObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps): # Removed initial_history and initial_action
        """
        Delays and stacks observations to create a history.

        Args:
            env: The environment to wrap.
            n_steps (int): The number of past observations to stack (lookback).
        """
        super().__init__(env)
        if n_steps <= 0:
             raise ValueError("n_steps must be a positive integer.")
        self.n_steps = n_steps

        # Determine the shape and dtype of a single observation
        single_obs_space = env.observation_space
        self.single_obs_shape = single_obs_space.shape
        self.dtype = single_obs_space.dtype

        # Initialize buffer with zeros matching the single observation shape/dtype
        self.history_buffer = deque(
            [np.zeros(self.single_obs_shape, dtype=self.dtype) for _ in range(n_steps)],
            maxlen=n_steps
        )
        self._has_reset = False # Track if reset has been called

        # Define the new observation space (stacked history)
        stacked_shape = (n_steps,) + self.single_obs_shape
        # Use low/high from original space if Box, otherwise use -inf/inf
        low = np.tile(single_obs_space.low, (n_steps, 1)) if isinstance(single_obs_space, gym.spaces.Box) else -np.inf
        high = np.tile(single_obs_space.high, (n_steps, 1)) if isinstance(single_obs_space, gym.spaces.Box) else np.inf
        # Ensure low/high have the correct final shape
        low = np.broadcast_to(low, stacked_shape)
        high = np.broadcast_to(high, stacked_shape)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=stacked_shape, dtype=self.dtype
        )

    def reset(self, **kwargs):
        """Resets the environment and the observation history buffer."""
        observation, info = self.env.reset(**kwargs)
        # Fill the buffer entirely with the first observation
        for _ in range(self.n_steps):
            self.history_buffer.append(observation) # deque automatically handles maxlen
        self._has_reset = True
        return np.stack(list(self.history_buffer), axis=0), info

    def observation(self, observation):
        """Adds the new observation to the buffer and returns the stacked history."""
        if not self._has_reset:
            # This might happen if step is called before reset, fill buffer as best effort
            print("Warning: DelayObservationWrapper.step() called before reset(). Filling buffer.")
            for _ in range(self.n_steps):
                self.history_buffer.append(observation)
            self._has_reset = True
        else:
            # Append new observation (deque automatically removes oldest)
            self.history_buffer.append(observation)

        return np.stack(list(self.history_buffer), axis=0)

    def get_history(self):
        """Returns the current observation history as a stacked numpy array."""
        return np.stack(list(self.history_buffer), axis=0)


# --- SelectObservationWrapper ---
# (Remove this class entirely if you are sure you won't use sensor selection)
class SelectObservationWrapper(gym.ObservationWrapper):
    def __init__(self, environment, selected_indices):
        super().__init__(environment)
        self.selected_indices = selected_indices
        original_space = self.env.observation_space
        if isinstance(original_space, gym.spaces.Box):
            low = original_space.low[self.selected_indices]
            high = original_space.high[self.selected_indices]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=original_space.dtype)
        # ... (handle other space types if necessary) ...
        else:
            raise NotImplementedError("SelectObservationWrapper currently only supports Box spaces.")

    def observation(self, observation):
        # Assuming observation is a numpy array for Box spaces
        return observation[self.selected_indices]


# --- CustomRescaleAction ---
class CustomRescaleAction(gym.Wrapper):
    def __init__(self, env, min_action, max_action):
        super(CustomRescaleAction, self).__init__(env)
        # Ensure min/max_action are numpy arrays for consistent operations
        self.min_action = np.asarray(min_action)
        self.max_action = np.asarray(max_action)

        if not hasattr(env.action_space, 'shape'):
            raise TypeError("Underlying environment's action space must have a 'shape' attribute.")
        if self.min_action.shape != env.action_space.shape:
            raise ValueError(f"min_action shape {self.min_action.shape} != env action space shape {env.action_space.shape}")
        if self.max_action.shape != env.action_space.shape:
             raise ValueError(f"max_action shape {self.max_action.shape} != env action space shape {env.action_space.shape}")

        # New action space is always [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def step(self, action):
        # Clip action just in case, then rescale
        action = np.clip(np.asarray(action), -1.0, 1.0)
        scaled_action = self.min_action + (0.5 * (action + 1.0) * (self.max_action - self.min_action))
        # Ensure output shape matches original env expectation if needed (e.g., squeeze if was scalar)
        # scaled_action = scaled_action.reshape(self.env.action_space.shape)
        return self.env.step(scaled_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)