import os

import gymnasium as gym
from PIL import Image

env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
               params={"generation": "random", "environment_seed": 127, "hide_state_variables": True, "img_obs": True,
                       "simple_reset": True, "img_dim": 400},
               render_mode="rgb_array",
               )

num_images = 100

dir = "dataset/waterworld"
os.makedirs(dir, exist_ok=True)

for i in range(num_images):
    obs, info = env.reset(seed=i)
    img = Image.fromarray(obs)
    img.save(f"{dir}/img_{i}.png") 