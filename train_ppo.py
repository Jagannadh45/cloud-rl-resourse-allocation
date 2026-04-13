from stable_baselines3 import PPO
from env.cluster_env import CloudClusterEnv


env = CloudClusterEnv(nodes=3)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/"
)

print("==== Training Started ====")

model.learn(total_timesteps=100000)

model.save("cloud_cluster_model")

print("==== Training Completed ====")
