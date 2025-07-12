import gym_subgoal_automata
import gymnasium as gym

from gym_subgoal_automata.utils.visualisation import interactive_visualisation_pygame, VisualisationLogicWrapper

seed = 1

env = gym.make("VisualMinecraftGemDoorEnv-v0",
               params={"generation": "random", "environment_seed": 127, "hide_state_variables": True, "img_obs": True,
                       "simple_reset": False},
               render_mode="human",
               )

logic = VisualisationLogicWrapper(env, seed)

interactive_visualisation_pygame(logic, n_rows=1, n_cols=1)

