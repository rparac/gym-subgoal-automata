import collections
import itertools
import math
import time
import numpy as np
import random
import pygame
import torchvision
from PIL import Image
from gymnasium import spaces
from gym_subgoal_automata.utils import utils
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from gym_subgoal_automata.envs.base.base_env import BaseEnv


class VisualMinecraftActions:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class VisualMinecraftObservations:
    PICKAXE = "P"
    LAVA = "L"
    DOOR = "D"
    GEM = "G"
    EMPTY = "E"


class VisualMinecraftEnv(BaseEnv):
    """
    The Visual Minecraft environment based on NeuralRewardMachines
    from "Neural Reward Machines for Expressively Specifying Reward Functions"

    Description:
    It consists of a 4x4 grid containing different objects:
    - Pickaxe (P): Tool that can be collected
    - Lava (L): Dangerous area to avoid
    - Door (D): Exit that can be reached
    - Gem (G): Valuable item to collect
    - Empty (E): Free space to move

    The agent starts at position (0,0) and can move in four directions.
    Different tasks can be defined using LTL formulas that specify
    reward functions based on the sequence of observations.

    Rewards:
    Different tasks are defined using LTL formulas that create reward machines.
    Examples:
    - Collect gem then reach door: G & F D
    - Avoid lava while collecting pickaxe: !L U P
    - Complex sequences with multiple objectives

    Actions:
    - 0: up
    - 1: right
    - 2: down
    - 3: left

    Acknowledgments:
    Based on the NeuralRewardMachines implementation:
    https://github.com/KRLGroup/NeuralRewardMachines
    """
    # Object symbols
    SYMBOL_DOOR = "door"
    SYMBOL_PICKAXE = "pickaxe"
    SYMBOL_GEM = "gem"
    SYMBOL_LAVA = "lava"
    SYMBOL_EMPTY = "empty"
    
    # LTL formulas
    LTL_GEM_DOOR = "gem & F door"
    LTL_AVOID_LAVA_PICKAXE = "!lava U pickaxe"
    LTL_COMPLEX_TASK = "pickaxe & F gem & F door"
    
    # Default parameters
    DEFAULT_SIZE = 4
    DEFAULT_MAX_STEPS = 50
    DEFAULT_WINDOW_SIZE = 512
    DEFAULT_IMG_DIM = 64
    
    # Rewards
    REWARD_STEP = -1
    REWARD_ACCEPT_STATE = 10
    REWARD_REJECT_STATE = -10
    REWARD_FINAL_SUCCESS = 100
    REWARD_LAVA_DEATH = -100
    
    # Colors for rendering
    COLOR_AGENT = (0, 0, 255)  # Blue
    COLOR_BORDER = (0, 0, 0)   # Black
    COLOR_BACKGROUND = (255, 255, 255)  # White
    
    RENDERING_COLORS = {
        VisualMinecraftObservations.PICKAXE: (139, 69, 19),  # Brown
        VisualMinecraftObservations.LAVA: (255, 0, 0),       # Red
        VisualMinecraftObservations.DOOR: (0, 255, 0),       # Green
        VisualMinecraftObservations.GEM: (255, 215, 0),      # Gold
        VisualMinecraftObservations.EMPTY: (255, 255, 255),  # White
    }

    # when a seed is fixed, a derived seed is used every time the environment is restarted,
    # helps with reproducibility while generalizing to different starting positions
    RANDOM_RESTART = "random_restart"

    def __init__(self, params, ltl_formula=None, render_mode=None):
        super().__init__(params, render_mode)

        self.random_restart = utils.get_param(params, VisualMinecraftEnv.RANDOM_RESTART, True)
        self.num_resets = 0

        # Grid parameters
        self.size = utils.get_param(params, "size", self.DEFAULT_SIZE)
        self.max_steps = utils.get_param(params, "max_steps", self.DEFAULT_MAX_STEPS)
        self.curr_step = 0

        # LTL formula for reward machine
        self.ltl_formula = ltl_formula or self.LTL_GEM_DOOR  # Default: eventually reach door
        self.automaton = self._create_automaton_from_ltl(self.ltl_formula)
        self.curr_automaton_state = 0

        # Grid layout - fixed positions for objects (same as NeuralRewardMachines)
        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([1, 1])
        self._door_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])

        # Agent position
        self._agent_location = np.array([0, 0])

        # Object states (collected/available)
        self._gem_collected = False
        self._pickaxe_collected = False
        self._door_reached = False

        # Action space and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.size-1, shape=(2 + self.automaton.get_num_states(),), dtype=np.float32
        )

        # Rendering attributes
        self.game_display = None
        self.clock = None
        self.window_size = utils.get_param(params, "window_size", self.DEFAULT_WINDOW_SIZE)

        # Image observation support
        self._image_obs = utils.get_param(params, "img_obs", False)
        self._output_img_dim = utils.get_param(params, "img_dim", self.DEFAULT_IMG_DIM)
        if self._image_obs:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((self._output_img_dim, self._output_img_dim)),
                torchvision.transforms.ToPILImage(),
            ])
            self._resize_fn = transforms
            pygame.init()
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._output_img_dim, self._output_img_dim, 3), dtype=np.uint8
            )

    def _create_automaton_from_ltl(self, ltl_formula):
        """
        Create a simple automaton from LTL formula.
        This is a simplified version - in practice you'd use a proper LTL parser.
        """
        automaton = SubgoalAutomaton()
        
        # For now, create a simple automaton for common patterns
        if ltl_formula == self.LTL_GEM_DOOR:  # Eventually reach door
            automaton.add_state("s0")
            automaton.add_state("s1")
            automaton.set_initial_state("s0")
            automaton.set_accept_state("s1")
            
            # Add transitions
            automaton.add_edge("s0", "s1", [self.SYMBOL_DOOR])
            automaton.add_edge("s0", "s0", [self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            automaton.add_edge("s1", "s1", [self.SYMBOL_DOOR, self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            
        elif ltl_formula == self.LTL_AVOID_LAVA_PICKAXE:  # Avoid lava until pickaxe
            automaton.add_state("s0")
            automaton.add_state("s1")
            automaton.add_state("s2")
            automaton.set_initial_state("s0")
            automaton.set_accept_state("s1")
            automaton.set_reject_state("s2")
            
            # Add transitions
            automaton.add_edge("s0", "s1", [self.SYMBOL_PICKAXE])
            automaton.add_edge("s0", "s2", [self.SYMBOL_LAVA])
            automaton.add_edge("s0", "s0", [self.SYMBOL_GEM, self.SYMBOL_DOOR, self.SYMBOL_EMPTY])
            automaton.add_edge("s1", "s1", [self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_DOOR, self.SYMBOL_EMPTY, self.SYMBOL_LAVA])
            automaton.add_edge("s2", "s2", [self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_DOOR, self.SYMBOL_EMPTY, self.SYMBOL_LAVA])
            
        elif ltl_formula == self.LTL_COMPLEX_TASK:  # Collect pickaxe, then gem, then reach door
            automaton.add_state("s0")
            automaton.add_state("s1")
            automaton.add_state("s2")
            automaton.add_state("s3")
            automaton.set_initial_state("s0")
            automaton.set_accept_state("s3")
            
            # Add transitions
            automaton.add_edge("s0", "s1", [self.SYMBOL_PICKAXE])
            automaton.add_edge("s0", "s0", [self.SYMBOL_GEM, self.SYMBOL_DOOR, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            automaton.add_edge("s1", "s2", [self.SYMBOL_GEM])
            automaton.add_edge("s1", "s1", [self.SYMBOL_PICKAXE, self.SYMBOL_DOOR, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            automaton.add_edge("s2", "s3", [self.SYMBOL_DOOR])
            automaton.add_edge("s2", "s2", [self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            automaton.add_edge("s3", "s3", [self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_DOOR, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
            
        else:
            # Default automaton - just accept everything
            automaton.add_state("s0")
            automaton.set_initial_state("s0")
            automaton.set_accept_state("s0")
            automaton.add_edge("s0", "s0", [self.SYMBOL_DOOR, self.SYMBOL_PICKAXE, self.SYMBOL_GEM, self.SYMBOL_LAVA, self.SYMBOL_EMPTY])
        
        return automaton

    def _get_current_symbol(self):
        """Get the current symbol based on agent location"""
        if (self._agent_location == self._door_location).all():
            return self.SYMBOL_DOOR
        if (self._agent_location == self._pickaxe_location).all() and not self._pickaxe_collected:
            return self.SYMBOL_PICKAXE
        if (self._agent_location == self._gem_location).all() and not self._gem_collected:
            return self.SYMBOL_GEM
        if (self._agent_location == self._lava_location).all():
            return self.SYMBOL_LAVA
        return self.SYMBOL_EMPTY

    def _update_object_states(self):
        """Update object states based on agent position"""
        if (self._agent_location == self._pickaxe_location).all() and not self._pickaxe_collected:
            self._pickaxe_collected = True
        if (self._agent_location == self._gem_location).all() and not self._gem_collected:
            self._gem_collected = True
        if (self._agent_location == self._door_location).all():
            self._door_reached = True

    def step(self, action):
        """Execute one step in the environment"""
        self.curr_step += 1
        reward = self.REWARD_STEP  # Small negative reward per step
        done = False
        truncated = False

        # Get current symbol before movement
        current_symbol = self._get_current_symbol()

        # Move agent
        if action == VisualMinecraftActions.UP:
            direction = np.array([0, -1])
        elif action == VisualMinecraftActions.RIGHT:
            direction = np.array([1, 0])
        elif action == VisualMinecraftActions.DOWN:
            direction = np.array([0, 1])
        elif action == VisualMinecraftActions.LEFT:
            direction = np.array([-1, 0])
        else:
            raise ValueError(f"Invalid action: {action}")

        # Update agent position (with bounds checking)
        new_location = self._agent_location + direction
        self._agent_location = np.clip(new_location, 0, self.size - 1)

        # Update object states
        self._update_object_states()

        # Get new symbol and update automaton
        new_symbol = self._get_current_symbol()
        old_state = self.curr_automaton_state
        self.curr_automaton_state = self.automaton.get_next_state(
            self.curr_automaton_state, new_symbol
        )

        # Calculate reward based on automaton transition
        if self.curr_automaton_state != old_state:
            # Simple reward: +10 for reaching accepting state, -10 for reaching rejecting state
            if self.automaton.is_accept_state(self.curr_automaton_state):
                reward = self.REWARD_ACCEPT_STATE
            elif self.automaton.is_reject_state(self.curr_automaton_state):
                reward = self.REWARD_REJECT_STATE
            else:
                reward = 0

        # Check termination conditions
        if self.automaton.is_accept_state(self.curr_automaton_state):
            reward = self.REWARD_FINAL_SUCCESS  # Large positive reward for reaching accepting state
            done = True
        elif new_symbol == self.SYMBOL_LAVA:
            reward = self.REWARD_LAVA_DEATH  # Large negative reward for hitting lava
            done = True

        # Check timeout
        if self.curr_step >= self.max_steps:
            truncated = True

        # Get observation
        observation = self.get_observations()

        # Get info
        info = {
            "automaton_state": self.curr_automaton_state,
            "symbol": new_symbol,
            "step": self.curr_step
        }

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed, options)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.curr_step = 0
        self.curr_automaton_state = self.automaton.initial_state
        self._agent_location = np.array([0, 0])
        self._gem_collected = False
        self._pickaxe_collected = False
        self._door_reached = False

        observation = self.get_observations()
        info = {
            "automaton_state": self.curr_automaton_state,
            "symbol": self._get_current_symbol(),
            "step": self.curr_step
        }

        return observation, info

    def get_observations(self):
        """Get current observation"""
        if self._image_obs:
            return self._get_image()
        else:
            return self._get_features()

    def _get_features(self):
        """Get feature-based observation"""
        # Agent position + automaton state
        obs = np.array(list(self._agent_location), dtype=np.float32)
        
        # Add automaton state as one-hot encoding
        state_idx = self.automaton.get_state_id(self.curr_automaton_state)
        automaton_state_onehot = np.zeros(self.automaton.get_num_states(), dtype=np.float32)
        automaton_state_onehot[state_idx] = 1.0
        
        return np.concatenate([obs, automaton_state_onehot])

    def _get_image(self):
        """Get image-based observation"""
        if self.game_display is None:
            pygame.init()
            self.game_display = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Create image
        image = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        cell_size = self.window_size // self.size

        # Draw grid
        for i in range(self.size):
            for j in range(self.size):
                x, y = i * cell_size, j * cell_size
                
                # Determine cell content
                pos = np.array([i, j])
                if (pos == self._agent_location).all():
                    color = self.COLOR_AGENT  # Blue for agent
                elif (pos == self._gem_location).all() and not self._gem_collected:
                    color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.GEM]
                elif (pos == self._pickaxe_location).all() and not self._pickaxe_collected:
                    color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.PICKAXE]
                elif (pos == self._door_location).all():
                    color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.DOOR]
                elif (pos == self._lava_location).all():
                    color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.LAVA]
                else:
                    color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.EMPTY]

                # Fill cell
                image[y:y+cell_size, x:x+cell_size] = color

        # Resize image
        image = Image.fromarray(image)
        image = self._resize_fn(image)
        return np.array(image)

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.game_display is None:
                pygame.init()
                self.game_display = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()

            # Clear screen
            self.game_display.fill(self.COLOR_BACKGROUND)
            cell_size = self.window_size // self.size

            # Draw grid
            for i in range(self.size):
                for j in range(self.size):
                    x, y = i * cell_size, j * cell_size
                    
                    # Determine cell content
                    pos = np.array([i, j])
                    if (pos == self._agent_location).all():
                        color = self.COLOR_AGENT  # Blue for agent
                    elif (pos == self._gem_location).all() and not self._gem_collected:
                        color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.GEM]
                    elif (pos == self._pickaxe_location).all() and not self._pickaxe_collected:
                        color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.PICKAXE]
                    elif (pos == self._door_location).all():
                        color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.DOOR]
                    elif (pos == self._lava_location).all():
                        color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.LAVA]
                    else:
                        color = VisualMinecraftEnv.RENDERING_COLORS[VisualMinecraftObservations.EMPTY]

                    # Draw cell
                    pygame.draw.rect(self.game_display, color, (x, y, cell_size, cell_size))
                    pygame.draw.rect(self.game_display, self.COLOR_BORDER, (x, y, cell_size, cell_size), 1)

            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(10)

        elif self.render_mode == "rgb_array":
            return self._get_image()

    def close(self):
        """Close the environment"""
        if self.game_display is not None:
            pygame.quit()

    def is_terminal(self):
        """Check if the episode is terminal"""
        return self.automaton.is_accept_state(self.curr_automaton_state) or \
               self._get_current_symbol() == self.SYMBOL_LAVA

    def get_observables(self):
        """Get list of observable symbols"""
        return [self.SYMBOL_PICKAXE, self.SYMBOL_LAVA, self.SYMBOL_DOOR, self.SYMBOL_GEM, self.SYMBOL_EMPTY]

    def get_restricted_observables(self):
        """Get restricted observables (same as full observables for this env)"""
        return self.get_observables()

    def get_automaton(self):
        """Get the subgoal automaton"""
        return self.automaton

    def play(self):
        """Interactive play mode"""
        self.reset()
        done = False
        
        while not done:
            self.render()
            
            # Get user input
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = VisualMinecraftActions.UP
                    elif event.key == pygame.K_RIGHT:
                        action = VisualMinecraftActions.RIGHT
                    elif event.key == pygame.K_DOWN:
                        action = VisualMinecraftActions.DOWN
                    elif event.key == pygame.K_LEFT:
                        action = VisualMinecraftActions.LEFT
                    elif event.key == pygame.K_ESCAPE:
                        return

            if action is not None:
                observation, reward, done, truncated, info = self.step(action)
                print(f"Action: {action}, Reward: {reward}, State: {info['automaton_state']}, Symbol: {info['symbol']}")

        print("Episode finished!")
        self.render()
        time.sleep(2)


# Specific environment variants
class VisualMinecraftGemDoorEnv(VisualMinecraftEnv):
    """VisualMinecraft environment: collect gem then reach door"""
    
    def __init__(self, params=None, render_mode=None):
        ltl_formula = self.LTL_GEM_DOOR  # Eventually reach door
        super().__init__(params, ltl_formula, render_mode)


class VisualMinecraftAvoidLavaPickaxeEnv(VisualMinecraftEnv):
    """VisualMinecraft environment: avoid lava until pickaxe"""
    
    def __init__(self, params=None, render_mode=None):
        ltl_formula = self.LTL_AVOID_LAVA_PICKAXE  # Avoid lava until pickaxe
        super().__init__(params, ltl_formula, render_mode)


class VisualMinecraftComplexEnv(VisualMinecraftEnv):
    """VisualMinecraft environment: complex task with multiple objectives"""
    
    def __init__(self, params=None, render_mode=None):
        # More complex LTL formula: collect pickaxe, then gem, then reach door while avoiding lava
        ltl_formula = self.LTL_COMPLEX_TASK
        super().__init__(params, ltl_formula, render_mode) 