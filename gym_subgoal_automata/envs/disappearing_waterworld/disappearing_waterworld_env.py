import itertools
import math
import random
import time

import numpy as np
import pygame
import torchvision
from gymnasium import spaces

from gym_subgoal_automata.envs.base.base_env import BaseEnv
from gym_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldObservations, WaterWorldActions, BallSequence, \
    Ball, BallAgent
from gym_subgoal_automata.utils import utils
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton


class DisappearingWaterWorldEnv(BaseEnv):
    """
    The Water World environment
    from "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning"
    by Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Valenzano and Sheila A. McIlraith.

    Description:
    It consists of a 2D box containing 12 balls of different colors (2 balls per color). Each ball moves at a constant
    speed in a given direction and bounces when it collides with a wall. The agent is a white ball that can change its
    velocity in any of the four cardinal directions.

    Rewards:
    Different tasks (subclassed below) are defined in this environment. All of them are goal-oriented, i.e., provide
    a reward of 1 when a certain goal is achieved and 0 otherwise. The goal always consists of touching a sequence of
    balls in a specific order. We describe some examples:
        - WaterWorldRedGreenEnv: touch a red ball, then a green ball.
        - WaterWorldRedGreenAndMagentaYellowEnv: touch a red ball then a green ball, and touch a magenta ball then a
                                                 yellow ball. Note that the two sequences can be interleaved.
        - WaterWorldRedGreenBlueStrictEnv: touch a red ball, then a green ball. No other balls can be touched, so the
                                           agent has to avoid touching nothing but a red ball first, and nothing but a
                                           green ball afterwards.

    Actions:
    - 0: up
    - 1: down
    - 2: left
    - 3: right
    - 4: none

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    RENDERING_COLORS = {"A": (0, 0, 0),
                        WaterWorldObservations.RED: (255, 0, 0),
                        WaterWorldObservations.GREEN: (0, 255, 0),
                        WaterWorldObservations.BLUE: (0, 0, 255),
                        WaterWorldObservations.YELLOW: (255, 255, 0),
                        WaterWorldObservations.CYAN: (0, 255, 255),
                        WaterWorldObservations.MAGENTA: (255, 0, 255)
                        }

    # when a seed is fixed, a derived seed is used every time the environment is restarted,
    # helps with reproducibility while generalizing to different starting positions
    RANDOM_RESTART = "random_restart"

    def __init__(self, params, sequences, obs_to_avoid=None, render_mode=None):
        super().__init__(params, render_mode)

        self.random_restart = utils.get_param(params, DisappearingWaterWorldEnv.RANDOM_RESTART, True)
        self.num_resets = 0

        # check input sequence
        self._check_sequences(sequences)

        # sequences of balls that have to be touched
        self.sequences = sequences
        self.state = None  # current index in each sequence
        self.last_strict_obs = None  # last thing observed (used only for strict sequences)

        # set of observables to avoid seeing at anytime (only when the sequence is not strict)
        self.obs_to_avoid = obs_to_avoid

        # parameters
        self.max_x = utils.get_param(params, "max_x", 400)
        self.max_y = utils.get_param(params, "max_y", 400)
        self.ball_num_colors = len(self.get_observables())
        self.ball_radius = utils.get_param(params, "ball_radius", 15)
        self.ball_velocity = utils.get_param(params, "ball_velocity", 30)
        self.ball_num_per_color = utils.get_param(params, "ball_num_per_color", 2)
        self.use_velocities = utils.get_param(params, "use_velocities", True)
        self.agent_vel_delta = self.ball_velocity
        self.agent_vel_max = 3 * self.ball_velocity

        # agent ball and other balls to avoid or touch
        self.agent = None
        self.balls = []

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.max_x, self.max_y, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(5)

        # rendering attributes
        self.game_display = None
        self.clock = None

        # Image observation
        self._output_img_dim = utils.get_param(params, "img_dim", 64)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self._output_img_dim, self._output_img_dim)),
            torchvision.transforms.ToPILImage(),
        ])
        self._resize_fn = transforms
        pygame.init()

        # Allows colliding balls on reset
        self._simple_reset = utils.get_param(params, "simple_reset", False)

    @staticmethod
    def _check_sequences(sequences):
        for sequence in sequences:
            if sequence.is_strict and len(sequences) > 1:
                raise Exception("Error: Sequences containing one strict subsequence must only contain this item!")

    def _get_pos_vel_new_ball(self, random_gen):
        angle = random_gen.random() * 2 * math.pi

        if self.use_velocities:
            vel = self.ball_velocity * math.sin(angle), self.ball_velocity * math.cos(angle)
        else:
            vel = 0.0, 0.0

        while True:
            pos = 2 * self.ball_radius + random_gen.random() * (self.max_x - 4 * self.ball_radius), \
                  2 * self.ball_radius + random_gen.random() * (self.max_y - 4 * self.ball_radius)
            if not self._is_colliding(pos) and np.linalg.norm(self.agent.pos - np.array(pos),
                                                              ord=2) > 4 * self.ball_radius or self._simple_reset:
                break
        return pos, vel

    def _is_colliding(self, pos):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.pos - np.array(pos), ord=2) < 2 * self.ball_radius:
                return True
        return False

    def get_observations(self):
        ret = {"observations": {b.color for b in self._get_current_collisions()}}
        return ret

    def remove_balls(self):
        balls = self._get_current_collisions()
        for b in balls:
            self.balls.remove(b)


    def get_observables(self):
        return [WaterWorldObservations.RED, WaterWorldObservations.GREEN, WaterWorldObservations.BLUE,
                WaterWorldObservations.CYAN, WaterWorldObservations.MAGENTA, WaterWorldObservations.YELLOW]

    def get_restricted_observables(self):
        return self._get_symbols_from_sequence()

    def _get_current_collisions(self):
        return self._get_current_ball_collisions(self.agent)

    def _get_current_ball_collisions(self, b):
        collisions = set()
        for other_b in self.balls:
            if other_b != b and b.is_colliding(other_b):
                collisions.add(other_b)
        return collisions

    def is_terminal(self):
        return self.is_game_over

    def step(self, action, elapsed_time=0.1):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.is_game_over:
            observations = self.get_observations()
            self.remove_balls()
            return self._get_features(), 0.0, True, False, observations

        # updating the agents velocity
        self.agent.step(action)
        balls_all = [self.agent] + self.balls
        max_x, max_y = self.max_x, self.max_y

        # updating position
        for b in balls_all:
            b.update_position(elapsed_time)

        # handling collisions
        for i in range(len(balls_all)):
            b = balls_all[i]
            is_colliding = len(self._get_current_ball_collisions(b)) > 0

            # walls or another balls
            if b.pos[0] - b.radius < 0 or b.pos[0] + b.radius > max_x:
                # place ball against edge
                if b.pos[0] - b.radius < 0:
                    b.pos[0] = b.radius
                else:
                    b.pos[0] = max_x - b.radius
                # reverse direction
                b.vel *= np.array([-1.0, 1.0])
            if b.pos[1] - b.radius < 0 or b.pos[1] + b.radius > max_y:
                # place ball against edge
                if b.pos[1] - b.radius < 0:
                    b.pos[1] = b.radius
                else:
                    b.pos[1] = max_y - b.radius
                # reverse direction
                b.vel *= np.array([1.0, -1.0])

            if is_colliding and b != self.agent:
                # Reverse direction for ball collisions; the agent "captures" the
                #  ball, so it's direction doesn't reverse
                b.vel *= np.array([-1.0, -1.0])

        observations = self.get_observations()
        reward, is_done = self._step(observations)

        if is_done:
            self.is_game_over = True

        self.remove_balls()

        return self._get_features(), reward, is_done, False, observations

    def _step(self, observations):
        reached_terminal_state = self._update_state(observations)
        if reached_terminal_state:
            return 0.0, True

        if self.is_goal_achieved():
            return 1.0, True

        return 0.0, False

    def _update_state(self, observations):
        for i in range(0, len(self.sequences)):
            current_index = self.state[i]
            sequence = self.sequences[i]
            if sequence.is_strict:
                if len(observations) == 0:
                    self.last_strict_obs = None
                elif len(observations) == 1:
                    if not self._is_subgoal_in_observation(self.last_strict_obs, observations):
                        if self._is_subgoal_in_observation(sequence.sequence[current_index], observations):
                            self.last_strict_obs = sequence.sequence[current_index]
                            self.state[i] = current_index + 1
                        else:
                            return True
                else:
                    return True
            else:
                if self.obs_to_avoid is not None and self._contains_observable_to_avoid(observations):
                    return True
                while current_index < len(sequence.sequence) and self._is_subgoal_in_observation(
                        sequence.sequence[current_index], observations):
                    current_index += 1
                self.state[i] = current_index

        return False

    def _contains_observable_to_avoid(self, observation):
        for o in observation['observations']:
            if o in self.obs_to_avoid:
                return True
        return False

    @staticmethod
    def _is_subgoal_in_observation(subgoal, observation):
        for s in subgoal:
            if (isinstance(observation, dict) and s not in observation["observations"]) \
                    or (isinstance(observation, tuple) and s not in observation):
                return False
        return True

    def is_goal_achieved(self):
        # all sequences have been observed
        for i in range(0, len(self.sequences)):
            sequence = self.sequences[i].sequence
            if self.state[i] != len(sequence):
                return False
        return True

    def _get_image(self):
        curr_render_mode = self.render_mode
        self.render_mode = "rgb_array"
        out = self.render()
        self.render_mode = curr_render_mode

        # img = self._resize_fn(out)
        img = out
        return np.array(img)

    def _get_features(self):
        # We assume the observation is image as balls disappear
        return self._get_image()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        self.last_strict_obs = None

        if self.random_restart and self.seed is not None:
            seed = self.seed + self.num_resets
            self.num_resets += 1
        else:
            seed = self.seed
        random_gen = random.Random(seed)

        # adding the agent
        pos_a = [2 * self.ball_radius + random_gen.random() * (self.max_x - 4 * self.ball_radius),
                 2 * self.ball_radius + random_gen.random() * (self.max_y - 4 * self.ball_radius)]
        self.agent = BallAgent("A", self.ball_radius, pos_a, [0.0, 0.0], self.agent_vel_delta, self.agent_vel_max)

        # adding the balls
        self.balls = []
        colors = self.get_observables()
        for c in range(self.ball_num_colors):
            for _ in range(self.ball_num_per_color):
                color = colors[c]
                pos, vel = self._get_pos_vel_new_ball(random_gen)
                ball = Ball(color, self.ball_radius, pos, vel)
                self.balls.append(ball)

        # reset current index in each sequence
        self.state = [0] * len(self.sequences)

        observations = self.get_observations()
        self.remove_balls()
        return self._get_features(), observations

    def render(self):
        if self.game_display is None and self.render_mode == "human":
            pygame.display.set_caption("Water World")
            self.game_display = pygame.display.set_mode((self.max_x, self.max_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.max_x, self.max_y))

        # printing image
        canvas.fill((255, 255, 255))
        for ball in self.balls:
            self._render_ball(canvas, ball, 0)
        self._render_ball(canvas, self.agent, 3)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.game_display.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            render_fps = 4
            self.clock.tick(render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_ball(self, game_display, ball, thickness):
        pygame.draw.circle(game_display, DisappearingWaterWorldEnv.RENDERING_COLORS[ball.color],
                           self._get_ball_position(ball, self.max_y), ball.radius, thickness)

    @staticmethod
    def _get_ball_position(ball, max_y):
        return int(round(ball.pos[0])), int(max_y) - int(round(ball.pos[1]))

    def close(self):
        pygame.quit()

    def get_automaton(self):
        automaton = SubgoalAutomaton()

        automaton.set_initial_state("u0")
        automaton.set_accept_state("u_acc")

        if self._is_strict_sequence():
            self._add_strict_transitions_to_automaton(automaton)
            automaton.set_reject_state("u_rej")
        else:
            if self.obs_to_avoid is not None:
                automaton.set_reject_state("u_rej")
            self._add_transitions_to_automaton(automaton)

        return automaton

    def _is_strict_sequence(self):
        for sequence in self.sequences:
            if sequence.is_strict:
                return True
        return False

    def _add_strict_transitions_to_automaton(self, automaton):
        sequence = self.sequences[0].sequence
        state_counter = 0

        for i in range(len(sequence)):
            symbol = sequence[i]
            current_state = "u%d" % state_counter
            if i == len(sequence) - 1:
                next_state = "u_acc"
            else:
                next_state = "u%d" % (state_counter + 1)

            other_symbols = [x for x in self.get_observables() if x != symbol]

            automaton.add_edge(current_state, next_state, [symbol] + ["~" + x for x in other_symbols])
            for other_symbol in other_symbols:
                automaton.add_edge(current_state, "u_rej", [other_symbol])
            state_counter += 1

    def _add_transitions_to_automaton(self, automaton):
        symbols = self._get_symbols_from_sequence()
        seq_tuple = self._get_sequence_tuple_from_ball_sequence()

        seq_tuples_to_states = {}
        current_state_id = 0

        queue = [seq_tuple]
        checked_tuples = set()

        while len(queue) > 0:
            seq_tuple = queue.pop(0)
            checked_tuples.add(seq_tuple)

            derived_to_transitions = {}

            # test all possible assignments of symbols to the current sequence and derive new sequences, which
            # correspond to all the possible children in the automaton for that state
            for l in range(len(symbols) + 1):
                for subset in itertools.combinations(symbols, l):
                    derived_tuple = self._get_derived_sequence_tuple_from_assignment(seq_tuple, subset)
                    if seq_tuple != derived_tuple:
                        # if the derivation is the same than the original sequence, discard it
                        if derived_tuple not in derived_to_transitions:
                            derived_to_transitions[derived_tuple] = []
                        derived_to_transitions[derived_tuple].append(subset)

                    # each tuple corresponds to a specific state
                    if derived_tuple not in seq_tuples_to_states:
                        if len(derived_tuple) == 0:
                            seq_tuples_to_states[derived_tuple] = "u_acc"
                        else:
                            seq_tuples_to_states[derived_tuple] = "u%d" % current_state_id
                            current_state_id += 1

                    # append the derived sequence to the queue to be analysed
                    if derived_tuple not in checked_tuples and derived_tuple not in queue:
                        queue.append(derived_tuple)

            # compress all the possible transitions to a derived sequence by checking which symbols never change their
            # value in these transitions (i.e., always appear as true or always appear as false)
            for derived_seq in derived_to_transitions:
                if derived_seq not in checked_tuples:
                    final_transition = []
                    for symbol in symbols:
                        if self._is_symbol_false_in_all_arrays(symbol, derived_to_transitions[derived_seq]):
                            final_transition.append("~" + symbol)
                        elif self._is_symbol_true_in_all_arrays(symbol, derived_to_transitions[derived_seq]):
                            final_transition.append(symbol)
                    if self.obs_to_avoid is not None:
                        for o in self.obs_to_avoid:
                            final_transition.append("~" + o)
                    to_state = seq_tuples_to_states[derived_seq]
                    automaton.add_edge(seq_tuples_to_states[seq_tuple], to_state, final_transition)

        if self.obs_to_avoid is not None:
            for automaton_state in [x for x in automaton.get_states() if not automaton.is_terminal_state(x)]:
                for o in self.obs_to_avoid:
                    automaton.add_edge(automaton_state, "u_rej", [o])

    def _get_symbols_from_sequence(self):
        symbols = set([])
        for sequence in self.sequences:
            for symbol in sequence.sequence:
                if type(symbol) is str:
                    symbols.add(symbol)
                elif type(symbol) is tuple:
                    for s in symbol:
                        symbols.add(s)
        return sorted(list(symbols))

    def _get_sequence_tuple_from_ball_sequence(self):
        tuple_seq = []
        for sequence in self.sequences:
            tuple_seq.append(tuple(sequence.sequence))
        return tuple(tuple_seq)

    def _get_derived_sequence_tuple_from_assignment(self, seq_tuple, assignment):
        derived_tuple = []

        # apply the assignment to each subsequence until it no longer holds
        for subsequence in seq_tuple:
            last_unsat = 0
            for i in range(len(subsequence)):
                if self._is_subgoal_in_observation(subsequence[i], assignment):
                    last_unsat += 1
                else:
                    break
            if last_unsat < len(subsequence):
                derived_tuple.append(subsequence[last_unsat:])

        return tuple(derived_tuple)

    @staticmethod
    def _is_symbol_false_in_all_arrays(symbol, arrays):
        for array in arrays:
            if symbol in array:
                return False
        return True

    @staticmethod
    def _is_symbol_true_in_all_arrays(symbol, arrays):
        for array in arrays:
            if symbol not in array:
                return False
        return True

    def play(self):
        self.reset()
        self.render()

        clock = pygame.time.Clock()

        t_previous = time.time()
        actions = set()

        total_reward = 0.0

        while not self.is_terminal():
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if WaterWorldActions.LEFT in actions and event.key == pygame.K_LEFT:
                        actions.remove(WaterWorldActions.LEFT)
                    elif WaterWorldActions.RIGHT in actions and event.key == pygame.K_RIGHT:
                        actions.remove(WaterWorldActions.RIGHT)
                    elif WaterWorldActions.UP in actions and event.key == pygame.K_UP:
                        actions.remove(WaterWorldActions.UP)
                    elif WaterWorldActions.DOWN in actions and event.key == pygame.K_DOWN:
                        actions.remove(WaterWorldActions.DOWN)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        actions.add(WaterWorldActions.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        actions.add(WaterWorldActions.RIGHT)
                    elif event.key == pygame.K_UP:
                        actions.add(WaterWorldActions.UP)
                    elif event.key == pygame.K_DOWN:
                        actions.add(WaterWorldActions.DOWN)

            t_current = time.time()
            t_delta = (t_current - t_previous)

            # getting the action
            if len(actions) == 0:
                a = WaterWorldActions.NONE
            else:
                a = random.choice(list(actions))

            # executing the action
            _, reward, is_done, truncated, _ = self.step(a, t_delta)
            total_reward += reward

            # printing image
            self.render()

            clock.tick(20)

            t_previous = t_current

        print("Game finished. Total reward: %.2f." % total_reward)

        self.close()


# For debugging purposes
class DisappearingWaterWorldRedEnv(DisappearingWaterWorldEnv):
    def __init__(self, params=None, render_mode=None):
        sequences = [BallSequence([WaterWorldObservations.RED], False)]
        super().__init__(params, sequences, render_mode=render_mode)


class DisappearingWaterWorldRedGreenEnv(DisappearingWaterWorldEnv):
    def __init__(self, params=None, render_mode=None):
        sequences = [BallSequence([WaterWorldObservations.RED, WaterWorldObservations.GREEN], False)]
        super().__init__(params, sequences, render_mode=render_mode)


class DisappearingWaterWorldRedGreenBlueEnv(DisappearingWaterWorldEnv):
    def __init__(self, params=None, render_mode=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE], False)]
        super().__init__(params, sequences, render_mode=render_mode)


class DisappearingWaterWorldRedGreenBlueCyanEnv(DisappearingWaterWorldEnv):
    def __init__(self, params=None, render_mode=None):
        sequences = [BallSequence([WaterWorldObservations.RED,
                                   WaterWorldObservations.GREEN,
                                   WaterWorldObservations.BLUE,
                                   WaterWorldObservations.CYAN], False)]
        super().__init__(params, sequences, render_mode=render_mode)
