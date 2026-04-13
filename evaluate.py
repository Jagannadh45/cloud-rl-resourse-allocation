from stable_baselines3 import PPO
from env.cluster_env import CloudClusterEnv
import numpy as np


env = CloudClusterEnv(nodes=3)
model = PPO.load("cloud_cluster_model")

episodes = 10
reward_history = []

for ep in range(episodes):

    obs = env.reset()
    total_reward = 0

    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        total_reward += reward

    reward_history.append(total_reward)

print("\n========== RESULTS ==========")
print("Average Episode Reward:", np.mean(reward_history))
print("=============================\n")
