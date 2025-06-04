import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from safe_rl.environments.delayed_cost_lander import DelayedCostLunarLander
from safe_rl.models.cost_model import CostModelEnsemble

def main():
    # Load models
    model = PPO.load("models/ppo_model")

    cost_model = CostModelEnsemble()
    cost_model.load("models")
    if not cost_model.trained:
        raise RuntimeError("Cost model is not trained. Please train it before evaluation.")
    print("Cost model loaded successfully.")
    print("PPO model loaded successfully.")

    delays = [5, 10, 20]
    results = {
        "standard": {delay: {"rewards": [], "costs": []} for delay in delays},
        "safe": {delay: {"rewards": [], "costs": []} for delay in delays}
    }

    for mode in ["standard", "safe"]:
    # for mode in ["standard"]:
        for delay in delays:
            print(f"Evaluating mode: {mode}, delay: {delay}")
            env = Monitor(DelayedCostLunarLander(gym.make("LunarLanderContinuous-v3"), delay=delay))
            for ep in range(30):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                ep_cost = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    if mode == "safe":
                        is_safe = cost_model.safe_action_mask(obs[None, :], action[None, :])[0]
                        if not is_safe:
                            action = np.zeros_like(action)  # fallback action
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    ep_cost += info.get("cost", 0.0)
                    done = terminated or truncated
                results[mode][delay]["rewards"].append(ep_reward)
                results[mode][delay]["costs"].append(ep_cost)

    # Prepare data for plotting
    mean_rewards_standard = [np.mean(results["standard"][d]["rewards"]) for d in delays]
    mean_costs_standard = [np.mean(results["standard"][d]["costs"]) for d in delays]

    mean_rewards_safe = [np.mean(results["safe"][d]["rewards"]) for d in delays]
    mean_costs_safe = [np.mean(results["safe"][d]["costs"]) for d in delays]

    x = np.arange(len(delays))  # positions for groups
    width = 0.35  # width of the bars

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Reward plot
    axs[0].bar(x - width/2, mean_rewards_standard, width, label='Standard')
    axs[0].bar(x + width/2, mean_rewards_safe, width, label='Safe')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([str(d) for d in delays])
    axs[0].set_xlabel('Cost Delay (steps)')
    axs[0].set_ylabel('Avg Episode Reward')
    axs[0].set_title('Reward vs Cost Delay')
    axs[0].legend()

    # Cost plot
    axs[1].bar(x - width/2, mean_costs_standard, width, label='Standard')
    axs[1].bar(x + width/2, mean_costs_safe, width, label='Safe')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([str(d) for d in delays])
    axs[1].set_xlabel('Cost Delay (steps)')
    axs[1].set_ylabel('Avg Episode Cost')
    axs[1].set_title('Cost vs Cost Delay')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("logs/eval_plot_by_delay.png")
    plt.show()


if __name__ == "__main__":
    main()
