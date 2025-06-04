import gymnasium as gym
import numpy as np
from collections import deque

class DelayedCostLunarLander(gym.Wrapper):
    def __init__(self, env, fuel_cost_threshold=0.5, delay=5):
        super().__init__(env)
        self.fuel_cost_threshold = fuel_cost_threshold
        self.total_fuel_cost = 0.0
        self.max_steps = 1000
        self.step_count = 0
        self.delay = delay  # number of steps to delay cost
        self.cost_buffer = deque(maxlen=delay)  # buffer to store immediate costs

    def reset(self, **kwargs):
        self.total_fuel_cost = 0.0
        self.step_count = 0
        self.cost_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_contact, right_contact = obs

        self.step_count += 1
        fuel_cost = np.linalg.norm(action)
        self.total_fuel_cost += fuel_cost

        # Immediate cost: crash penalty
        immediate_cost = 0.0
        if y_pos < 0.1 and abs(y_vel) > 1.0:
            immediate_cost = 5.0

        self.cost_buffer.append(immediate_cost)

        # If episode ends, compute final fuel penalty and flush remaining costs
        delayed_cost = 0.0
        if terminated or truncated or self.step_count >= self.max_steps:
            fuel_penalty = 1.0 if self.total_fuel_cost > self.fuel_cost_threshold * self.max_steps else 0.0
            self.cost_buffer.append(fuel_penalty)

            # Sum all remaining delayed costs into one final cost
            delayed_cost = sum(self.cost_buffer)
            self.cost_buffer.clear()
        elif len(self.cost_buffer) >= self.delay:
            delayed_cost = self.cost_buffer.popleft()
        else:
            delayed_cost = 0.0

        info["cost"] = delayed_cost
        return obs, reward, terminated, truncated, info
