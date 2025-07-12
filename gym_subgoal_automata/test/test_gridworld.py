#!/usr/bin/env python3
"""
Test script for the new GridWorld environment
"""

import sys
import os
import pytest
import numpy as np
from gymnasium import spaces

# Add the gym_subgoal_automata package to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_subgoal_automata.envs.gridworld.gridworld_env import GridWorldEnv, GridWorldActions


@pytest.fixture
def gridworld_env():
    """Fixture to create a GridWorld environment for testing"""
    # Create a simple concrete implementation for testing
    class TestGridWorldEnv(GridWorldEnv):
        def __init__(self, params=None, render_mode=None):
            super().__init__(params, render_mode)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.float32
            )
        
        def reset(self, seed=None, options=None):
            super().reset(seed, options)
            obs = np.zeros(4, dtype=np.float32)
            info = {"step": 0}
            return obs, info
        
        def step(self, action):
            obs = np.zeros(4, dtype=np.float32)
            reward = 0.0
            done = False
            truncated = False
            info = {"step": 1}
            return obs, reward, done, truncated, info
        
        def render(self):
            pass
        
        def close(self):
            pass
        
        def is_terminal(self):
            return False
        
        def get_observables(self):
            return []
        
        def get_restricted_observables(self):
            return []
        
        def get_automaton(self):
            return None
        
        def get_observations(self):
            return np.zeros(4, dtype=np.float32)
    
    env = TestGridWorldEnv()
    yield env
    env.close()


def test_gridworld_basic(gridworld_env):
    """Test basic GridWorld functionality"""
    # Test basic environment creation
    assert gridworld_env is not None
    assert hasattr(gridworld_env, 'reset')
    assert hasattr(gridworld_env, 'step')
    assert hasattr(gridworld_env, 'render')
    assert gridworld_env.action_space.n == 4
    assert len(gridworld_env.observation_space.shape) == 1
    
    # Test reset
    obs, info = gridworld_env.reset()
    assert obs is not None
    assert obs.shape == gridworld_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert 'step' in info
    assert info['step'] == 0
    
    # Test step
    action = 1  # Move right
    next_obs, reward, done, truncated, info = gridworld_env.step(action)
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert 'step' in info


def test_gridworld_actions():
    """Test GridWorld actions"""
    assert GridWorldActions.UP == 0
    assert GridWorldActions.DOWN == 1
    assert GridWorldActions.LEFT == 2
    assert GridWorldActions.RIGHT == 3


def test_gridworld_utility_methods():
    """Test GridWorld utility methods"""
    # Test get_state_id
    state_id = GridWorldEnv.get_state_id(4, [2, 2], [1, 0])
    assert isinstance(state_id, int)
    
    # Test get_one_hot_state
    one_hot = GridWorldEnv.get_one_hot_state(4, 2)
    assert one_hot.shape == (4,)
    assert one_hot.dtype == np.float32
    assert one_hot[2] == 1.0
    assert np.sum(one_hot) == 1.0


if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v"]) 