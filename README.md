# DelayedCostRL

A Gymnasium-compatible wrapper for RL environments that introduces **delayed safety costs** for use in **Safe Reinforcement Learning (Safe RL)** and **constrained optimization** settings.


---



## 🛠️ Installation

1. Install Gymnasium and Box2D dependencies:

   ```bash
   pip install gymnasium[box2d]
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/felipperoza/delayed-cost-lunarlander.git
   cd delayed-cost-lunarlander
   ```

3. Install as a package:

   ```bash
   pip install -e .
   ```

---

## 🧩 Usage

```python
import gymnasium as gym
from safe_rl.environments.delayed_cost_lander import DelayedCostLunarLander

env = DelayedCostLunarLander(
    gym.make("LunarLander-v3"),
    delay=5,
    fuel_cost_threshold=0.5
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Reward: {reward:.2f}, Delayed Cost: {info['cost']:.2f}")
```

---

## ⚙️ Parameters

| Argument              | Description                                                                 | Default |
|-----------------------|-----------------------------------------------------------------------------|---------|
| `delay`               | Number of steps to delay the cost feedback                                  | `5`     |
| `fuel_cost_threshold` | Fuel usage threshold (as a fraction of max steps); exceeding it triggers a final cost | `0.5`   |

---
