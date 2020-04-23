"""
This script simply runs the highway driving environment.

If sumo_ctrl = True, the standard SUMO driver model controls the ego vehicle.
Otherwise, random actions are taken.
"""

import numpy as np
import sys
sys.path.append('../src')
import parameters_simulation as p
from driving_env import Highway

p.sim_params['safety_check'] = False
sumo_ctrl = False

np.random.seed(13)
env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=True)

episode_rewards = []
episode_steps = []
for i in range(0, 100):
    np.random.seed(i)
    obs = env.reset(sumo_ctrl=sumo_ctrl)
    done = False
    episode_reward = 0
    step = 0
    while done is False:
        if not sumo_ctrl:
            action = np.random.randint(9)
            obs, reward, done, info = env.step(action)
        else:
            obs, reward, done, info = env.step(0, sumo_ctrl=True)
        episode_reward += reward
        step += 1

    episode_rewards.append(episode_reward)
    episode_steps.append(step)
    print("Episode: " + str(i))
    print("Episode steps: " + str(step))
    print("Episode reward: " + str(episode_reward))

print(episode_rewards)
print(episode_steps)
