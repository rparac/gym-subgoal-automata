from abc import ABC, abstractmethod
import gymnasium as gym
from gym_subgoal_automata.utils import utils


class BaseEnv(ABC, gym.Env):
    RANDOM_SEED_FIELD = "environment_seed"

    metadata = {
        "render_modes": ["human", "rgb_array"]
    }

    def __init__(self, params=None, render_mode=None):
        super().__init__()

        self.params = params
        self.is_game_over = False
        self.seed = utils.get_param(self.params, BaseEnv.RANDOM_SEED_FIELD)
        self.render_mode = render_mode

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_observables(self):
        pass

    @abstractmethod
    def get_restricted_observables(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def get_automaton(self):
        pass

    @abstractmethod
    def reset(self, seed=None, options=None):
        self.is_game_over = False
        return None, {}

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def play(self):
        pass
