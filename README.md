# Deep_RL_models
This folder contains minimalistic implementations of different RL-models
- **Minigrid_key_door**: Contains different architectures of agent to solve the minigrid envs
  - PPO
  - A2C
- **Meta-RL A3C**: Implementation of A3C to train a meta-RL agent to navigate in a maze
  - A3C_main_data_parallelization: Is a version of the A3C where the child nodes collect data and the main node compute the gradient   
- **Planning_A2C**: Implementation of an A2C agent that can decide when to plan in a 5x5 grid world