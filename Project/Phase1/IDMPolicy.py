import gymnasium as gym
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

env = MetaDriveEnv({
    "manual_control": False,
    "traffic_density": 0.0,
    "start_seed": 42,
    "map": "C",
    "agent_policy": IDMPolicy,
})


obs, _ = env.reset()
terminated = False

while not terminated:
    obs, reward, terminated, truncated, info = env.step([0,0.5])
    if terminated or truncated:
        env.reset()

    env.render(mode="topdown")

env.close()