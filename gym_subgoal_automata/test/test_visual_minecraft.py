#!/usr/bin/env python3
"""
Pytest tests for the VisualMinecraft environment
"""

import sys
import os
import pytest
import numpy as np

# Add the gym_subgoal_automata package to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_subgoal_automata.envs.visual_minecraft.visual_minecraft_env import (
    VisualMinecraftEnv, 
    VisualMinecraftGemDoorEnv, 
    VisualMinecraftAvoidLavaPickaxeEnv, 
    VisualMinecraftComplexEnv
)


@pytest.fixture
def visual_minecraft_env():
    """Fixture to create a VisualMinecraft environment for testing"""
    env = VisualMinecraftGemDoorEnv()
    yield env
    env.close()


def test_visual_minecraft_environment_creation(visual_minecraft_env):
    """Test that VisualMinecraft environment can be created successfully"""
    assert visual_minecraft_env is not None
    assert hasattr(visual_minecraft_env, 'reset')
    assert hasattr(visual_minecraft_env, 'step')
    assert hasattr(visual_minecraft_env, 'render')
    assert visual_minecraft_env.action_space.n == 4
    assert len(visual_minecraft_env.observation_space.shape) == 1


def test_visual_minecraft_reset(visual_minecraft_env):
    """Test that environment reset works correctly"""
    obs, info = visual_minecraft_env.reset()
    
    # Check observation
    assert obs is not None
    assert obs.shape == visual_minecraft_env.observation_space.shape
    assert obs.dtype == np.float32
    
    # Check info
    assert isinstance(info, dict)
    assert 'automaton_state' in info
    assert 'symbol' in info
    assert 'step' in info
    assert info['step'] == 0


def test_visual_minecraft_step(visual_minecraft_env):
    """Test that environment step works correctly"""
    obs, info = visual_minecraft_env.reset()
    
    # Take a step
    action = 1  # Move right
    next_obs, reward, done, truncated, info = visual_minecraft_env.step(action)
    
    # Verify step returned valid data
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert 'automaton_state' in info
    assert 'symbol' in info
    assert 'step' in info


def test_visual_minecraft_automaton(visual_minecraft_env):
    """Test that automaton is properly configured"""
    automaton = visual_minecraft_env.get_automaton()
    
    assert automaton is not None
    assert automaton.get_num_states() > 0
    assert automaton.get_initial_state() is not None
    assert automaton.accept_state is not None
    assert len(automaton.get_states()) > 0


def test_visual_minecraft_variants():
    """Test different VisualMinecraft variants can be created"""
    # Test GemDoor variant
    env1 = VisualMinecraftGemDoorEnv()
    assert env1 is not None
    env1.close()
    
    # Test AvoidLavaPickaxe variant
    env2 = VisualMinecraftAvoidLavaPickaxeEnv()
    assert env2 is not None
    env2.close()
    
    # Test Complex variant
    env3 = VisualMinecraftComplexEnv()
    assert env3 is not None
    env3.close()


def test_visual_minecraft_observations(visual_minecraft_env):
    """Test observation generation"""
    obs, info = visual_minecraft_env.reset()
    
    # Test feature observations
    features = visual_minecraft_env._get_features()
    assert features is not None
    assert len(features.shape) == 1
    assert features.shape[0] == 2 + visual_minecraft_env.automaton.get_num_states()
    
    # Check agent position
    agent_pos = features[:2]
    assert agent_pos.shape == (2,)
    assert agent_pos.dtype == np.float32
    
    # Check automaton state one-hot
    automaton_state = features[2:]
    assert automaton_state.shape == (visual_minecraft_env.automaton.get_num_states(),)
    assert automaton_state.dtype == np.float32
    assert np.sum(automaton_state) == 1.0  # Exactly one state should be active


def test_visual_minecraft_symbol_generation(visual_minecraft_env):
    """Test symbol generation based on agent position"""
    visual_minecraft_env.reset()
    
    # Test initial symbol (should be empty at position [0,0])
    symbol = visual_minecraft_env._get_current_symbol()
    assert symbol == 'empty'
    
    # Test symbol at gem location
    visual_minecraft_env._agent_location = np.array([0, 3])
    symbol = visual_minecraft_env._get_current_symbol()
    assert symbol == 'gem'
    
    # Test symbol at pickaxe location
    visual_minecraft_env._agent_location = np.array([1, 1])
    symbol = visual_minecraft_env._get_current_symbol()
    assert symbol == 'pickaxe'


def test_visual_minecraft_grid_layout(visual_minecraft_env):
    """Test that the grid layout matches NeuralRewardMachines"""
    # Check object positions (should match NeuralRewardMachines)
    expected_gem = np.array([0, 3])
    expected_pickaxe = np.array([1, 1])
    expected_door = np.array([3, 0])
    expected_lava = np.array([3, 3])
    
    assert (visual_minecraft_env._gem_location == expected_gem).all()
    assert (visual_minecraft_env._pickaxe_location == expected_pickaxe).all()
    assert (visual_minecraft_env._door_location == expected_door).all()
    assert (visual_minecraft_env._lava_location == expected_lava).all()


def test_visual_minecraft_object_states(visual_minecraft_env):
    """Test object state updates"""
    visual_minecraft_env.reset()
    
    # Initially objects should not be collected
    assert not visual_minecraft_env._gem_collected
    assert not visual_minecraft_env._pickaxe_collected
    assert not visual_minecraft_env._door_reached
    
    # Move to pickaxe location
    visual_minecraft_env._agent_location = np.array([1, 1])
    visual_minecraft_env._update_object_states()
    assert visual_minecraft_env._pickaxe_collected
    
    # Move to gem location
    visual_minecraft_env._agent_location = np.array([0, 3])
    visual_minecraft_env._update_object_states()
    assert visual_minecraft_env._gem_collected


def test_visual_minecraft_automaton_transitions(visual_minecraft_env):
    """Test automaton state transitions"""
    visual_minecraft_env.reset()
    
    initial_state = visual_minecraft_env.curr_automaton_state
    
    # Move to door location (should trigger transition to accepting state)
    visual_minecraft_env._agent_location = np.array([3, 0])
    visual_minecraft_env._update_object_states()
    
    new_symbol = visual_minecraft_env._get_current_symbol()
    new_state = visual_minecraft_env.automaton.get_next_state(initial_state, new_symbol)
    
    assert new_state != initial_state
    assert visual_minecraft_env.automaton.is_accept_state(new_state)


def test_visual_minecraft_termination_conditions(visual_minecraft_env):
    """Test episode termination conditions"""
    visual_minecraft_env.reset()
    
    # Test lava termination
    visual_minecraft_env._agent_location = np.array([3, 3])
    assert visual_minecraft_env.is_terminal()
    
    # Test accepting state termination
    visual_minecraft_env.reset()
    visual_minecraft_env.curr_automaton_state = visual_minecraft_env.automaton.accept_state
    assert visual_minecraft_env.is_terminal()


def test_plot_visual_minecraft_automata():
    """Test that plots the automata from different VisualMinecraft environment variants"""
    # Create environments for each variant
    envs = {
        "GemDoor": VisualMinecraftGemDoorEnv(),
        "AvoidLavaPickaxe": VisualMinecraftAvoidLavaPickaxeEnv(),
        "Complex": VisualMinecraftComplexEnv()
    }

    for idx, (name, env) in enumerate(envs.items()):
        automaton = env.get_automaton()
        automaton.plot(f"logs/test_visual_minecraft_automata_{name}", f"{name}.png")

    # Clean up
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v"]) 