import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import safe_rl
from safe_rl.environments.delayed_cost_lander import DelayedCostLunarLander
from safe_rl.models.cost_model import CostModelEnsemble


LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Train PPO
env = Monitor(DelayedCostLunarLander(gym.make("LunarLanderContinuous-v3")))
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
model.learn(total_timesteps=50000)
model.save(os.path.join(MODEL_DIR, "ppo_model"))

# Collect data
X_data, y_data = [], []
for ep in range(100):
    obs, _ = env.reset()
    done = False
    episode = []
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.append((obs.copy(), action.copy(), info.get("cost", 0.0)))
        obs = next_obs
        done = terminated or truncated
    for s, a, c in episode:
        X_data.append(np.concatenate([s, a]))
        y_data.append(c)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Train cost model
cost_model = CostModelEnsemble()
cost_model.fit(X_data, y_data)
cost_model.save(MODEL_DIR)
np.savez(os.path.join(MODEL_DIR, "training_data.npz"), X=X_data, y=y_data)