#!/usr/bin/env python3
"""
Pytest tests for generating dataset images from environments
"""

import os
import pytest
import gymnasium as gym
from PIL import Image


@pytest.fixture
def waterworld_env():
    """Fixture to create a WaterWorld environment for testing"""
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
                   params={
                       "generation": "random", 
                       "environment_seed": 127, 
                       "hide_state_variables": True, 
                       "img_obs": True,
                       "simple_reset": True, 
                       "img_dim": 400
                   },
                   render_mode="rgb_array",
                   )
    yield env
    env.close()


@pytest.fixture
def dataset_dir():
    """Fixture to create and clean up dataset directory"""
    dir_path = "dataset/waterworld"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    # Optional: clean up generated files after test
    # import shutil
    # if os.path.exists(dir_path):
    #     shutil.rmtree(dir_path)


def test_waterworld_environment_creation(waterworld_env):
    """Test that WaterWorld environment can be created successfully"""
    assert waterworld_env is not None
    assert hasattr(waterworld_env, 'reset')
    assert hasattr(waterworld_env, 'step')
    assert hasattr(waterworld_env, 'render')


def test_waterworld_reset_returns_image(waterworld_env):
    """Test that environment reset returns a valid image observation"""
    obs, info = waterworld_env.reset(seed=42)
    
    # Check that observation is a numpy array
    assert hasattr(obs, 'shape')
    assert len(obs.shape) == 3  # Height, width, channels
    assert obs.shape[2] == 3  # RGB channels
    
    # Check that info contains expected keys
    assert isinstance(info, dict)


def test_waterworld_image_generation(waterworld_env, dataset_dir):
    """Test generating multiple images from WaterWorld environment"""
    num_images = 10  # Reduced for faster testing
    
    generated_files = []
    
    for i in range(num_images):
        obs, info = waterworld_env.reset(seed=i)
        
        # Verify observation is valid
        assert obs is not None
        assert obs.shape[0] > 0 and obs.shape[1] > 0
        
        # Save image
        img = Image.fromarray(obs)
        file_path = f"{dataset_dir}/img_{i}.png"
        img.save(file_path)
        generated_files.append(file_path)
        
        # Verify file was created
        assert os.path.exists(file_path)
        
        # Verify file has content
        assert os.path.getsize(file_path) > 0
    
    # Verify all files were created
    assert len(generated_files) == num_images


def test_waterworld_image_consistency(waterworld_env):
    """Test that images have consistent shape when using the same seed"""
    seed = 123
    
    # Generate two images with the same seed
    obs1, _ = waterworld_env.reset(seed=seed)
    obs2, _ = waterworld_env.reset(seed=seed)
    
    # Images should have the same shape (content may vary due to environment randomness)
    assert obs1.shape == obs2.shape
    assert obs1.dtype == obs2.dtype


def test_waterworld_image_diversity(waterworld_env):
    """Test that images are different when using different seeds"""
    # Generate images with different seeds
    obs1, _ = waterworld_env.reset(seed=1)
    obs2, _ = waterworld_env.reset(seed=2)
    
    # Images should be different (not identical)
    assert not (obs1 == obs2).all()


def test_waterworld_step_functionality(waterworld_env):
    """Test that environment step function works correctly"""
    obs, info = waterworld_env.reset(seed=42)
    
    # Take a step
    action = 0  # Assuming 0 is a valid action
    next_obs, reward, done, truncated, info = waterworld_env.step(action)
    
    # Verify step returned valid data
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


if __name__ == "__main__":
    # Run the original image generation if called directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        # Original functionality for generating 100 images
        env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
                       params={"generation": "random", "environment_seed": 127, "hide_state_variables": True, "img_obs": True,
                               "simple_reset": True, "img_dim": 400},
                       render_mode="rgb_array",
                       )
        
        num_images = 100
        dir = "dataset/waterworld"
        os.makedirs(dir, exist_ok=True)
        
        print(f"Generating {num_images} images in {dir}...")
        
        for i in range(num_images):
            obs, info = env.reset(seed=i)
            img = Image.fromarray(obs)
            img.save(f"{dir}/img_{i}.png")
            
        print(f"Successfully generated {num_images} images")
        env.close()
    else:
        # Run pytest
        pytest.main([__file__, "-v"]) 