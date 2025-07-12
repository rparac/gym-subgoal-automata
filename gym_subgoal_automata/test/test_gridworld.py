#!/usr/bin/env python3
"""
Test script for the new GridWorld environment
"""

import sys
import os

# Add the gym_subgoal_automata package to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_gridworld_basic():
    """Test basic GridWorld functionality"""
    try:
        from gym_subgoal_automata.envs.gridworld.gridworld_env import GridWorldEnv, GridWorldGemDoorEnv
        
        # Test basic environment creation
        print("Testing GridWorld environment creation...")
        env = GridWorldGemDoorEnv()
        print(f"✓ Environment created successfully")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Test reset
        print("\nTesting environment reset...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial info: {info}")
        
        # Test step
        print("\nTesting environment step...")
        action = 1  # Move right
        obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        
        # Test automaton
        print("\nTesting automaton...")
        automaton = env.get_automaton()
        print(f"✓ Automaton retrieved")
        print(f"  Number of states: {automaton.get_num_states()}")
        print(f"  States: {automaton.get_states()}")
        print(f"  Initial state: {automaton.get_initial_state()}")
        print(f"  Accept state: {automaton.accept_state}")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gridworld_variants():
    """Test different GridWorld variants"""
    try:
        from gym_subgoal_automata.envs.gridworld.gridworld_env import (
            GridWorldGemDoorEnv, 
            GridWorldAvoidLavaPickaxeEnv, 
            GridWorldComplexEnv
        )
        
        print("\nTesting GridWorld variants...")
        
        # Test GemDoor variant
        env1 = GridWorldGemDoorEnv()
        print(f"✓ GridWorldGemDoorEnv created")
        
        # Test AvoidLavaPickaxe variant
        env2 = GridWorldAvoidLavaPickaxeEnv()
        print(f"✓ GridWorldAvoidLavaPickaxeEnv created")
        
        # Test Complex variant
        env3 = GridWorldComplexEnv()
        print(f"✓ GridWorldComplexEnv created")
        
        print("✓ All variants created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Variant test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gridworld_observations():
    """Test observation generation"""
    try:
        from gym_subgoal_automata.envs.gridworld.gridworld_env import GridWorldGemDoorEnv
        
        print("\nTesting observation generation...")
        env = GridWorldGemDoorEnv()
        obs, info = env.reset()
        
        # Test feature observations
        features = env._get_features()
        print(f"✓ Feature observations generated")
        print(f"  Feature shape: {features.shape}")
        print(f"  Agent position: {features[:2]}")
        print(f"  Automaton state one-hot: {features[2:]}")
        
        # Test symbol generation
        symbol = env._get_current_symbol()
        print(f"✓ Symbol generated: {symbol}")
        
        print("✓ Observation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Observation test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing GridWorld Environment")
    print("=" * 40)
    
    success = True
    success &= test_gridworld_basic()
    success &= test_gridworld_variants()
    success &= test_gridworld_observations()
    
    if success:
        print("\n" + "=" * 40)
        print("🎉 All tests passed! GridWorld environment is working correctly.")
    else:
        print("\n" + "=" * 40)
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 