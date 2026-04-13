# Advanced Reinforcement Learning for Cloud Resource Scheduling

## Overview
This project implements a PPO-based adaptive cloud scheduler that dynamically allocates CPU and memory resources across multiple compute nodes. The environment simulates workload arrivals and cluster-level scheduling decisions.

## Features
- Multi-node cloud cluster simulation
- Poisson workload generator
- SLA-aware reward function
- Load balancing penalty
- PPO reinforcement learning agent
- Tensorboard experiment tracking
- Docker reproducibility

## Algorithm
Proximal Policy Optimization (PPO)

## State Space
- Per-node CPU and memory utilization
- Incoming task CPU and memory demand
- Global queue backlog

## Reward Design
The reward function optimizes:
- Resource utilization
- Task latency
- SLA violations
- Load imbalance

## Training

Run training:

python train_ppo.py

## Evaluation

Run evaluation:

python evaluate.py

## Tensorboard

tensorboard --logdir logs

## Docker Execution

docker build -t cloud-rl-advanced .
docker run cloud-rl-advanced

## Conclusion
The trained PPO agent learns scheduling strategies that improve utilization while maintaining low task backlog and balanced cluster load.
