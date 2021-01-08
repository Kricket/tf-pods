import math

import numpy as np
from typing import Any, Tuple

from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput, PlayInput
from pod.game import Player
from pod.util import PodState, clean_angle
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from vec2 import ORIGIN, EPSILON

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

THRUST_VALUES = 11
ANGLE_VALUES = 51
MAX_ACTION = THRUST_VALUES * ANGLE_VALUES - 1

THRUST_INC = Constants.max_thrust() / (THRUST_VALUES - 1)
ANGLE_INC = Constants.max_turn() * 2 / (ANGLE_VALUES - 1)

def play_to_action(thrust: int, angle: float) -> int:
    """
    Given a legal play (angle/thrust), find the nearest discrete action
    """
    thrust_pct = thrust / Constants.max_thrust()
    angle_pct = (angle + Constants.max_turn()) / (2 * Constants.max_turn())
    thrust_idx = math.floor(thrust_pct * (THRUST_VALUES - 1))
    angle_idx = math.floor(angle_pct * (ANGLE_VALUES - 1))
    return math.floor(thrust_idx * ANGLE_VALUES + angle_idx)

def action_to_play(action: int) -> Tuple[int, float]:
    # An integer in [0, THRUST_VALUES - 1]
    thrust_idx = int(action / ANGLE_VALUES)
    # An integer in [0, ANGLE_VALUES - 1]
    angle_idx = action % ANGLE_VALUES
    return thrust_idx * THRUST_INC, angle_idx * ANGLE_INC - Constants.max_turn()


class QPodController(Controller):
    def __init__(self):
        self.play_output = PlayOutput()

    def set_play(self, action, pod: PodState):
        """
        Convert the action to a PlayInput
        """
        (thrust, rel_angle) = action_to_play(action.item())
        self.play_output.thrust = thrust

        real_angle = rel_angle + pod.angle
        real_dir = ORIGIN.rotate(real_angle) * 1000
        self.play_output.target = pod.pos + real_dir

    def play(self, pi: PlayInput) -> PlayOutput:
        return self.play_output


class QPodEnvironment(PyEnvironment):
    def __init__(self, board: PodBoard):
        super().__init__()

        # The action is a single integer representing an index into a list of discrete possible (action, thrust) values
        self._action_spec = array_spec.BoundedArraySpec(
            (),
            np.int32,
            minimum=0,
            maximum=MAX_ACTION)

        # The observation encodes the pod's state and next two checkpoints, seen from the pod's cockpit
        # (So we can omit the pod's position because it's always (0,0))
        self._observation_spec = array_spec.ArraySpec(shape=(6,), dtype=np.float)

        self._time_step_spec = ts.TimeStep(
            step_type=array_spec.ArraySpec(shape=(), dtype=np.int32, name='step_type'),
            reward=array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward'),
            discount=array_spec.ArraySpec(shape=(), dtype=np.float32, name='discount'),
            observation=self._observation_spec
        )

        self._board = board
        self._player = Player(QPodController())
        self._initial_state = self.get_state()
        self._episode_ended = False

    def get_state(self) -> Any:
        return self._player.pod.serialize()

    def set_state(self, state: Any) -> None:
        self._player.pod.deserialize(state)

    def get_info(self) -> Any:
        raise NotImplementedError("What is this?")

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self) -> ts.TimeStep:
        return self._time_step_spec

    def _reset(self):
        self.set_state(self._initial_state)
        self._episode_ended = False
        return ts.restart(self._to_observation())

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if self._player.pod.nextCheckId > 1 or self._pod.turns > 100:
            # That's enough for training...
            self._episode_ended = True
        else:
            # Play the given action
            self._player.controller.set_play(action, self._player.pod)
            self._player.step(self._board)

        if self._episode_ended:
            return ts.termination(self._to_observation(), self._get_reward())
        else:
            return ts.transition(self._to_observation(), reward = self._get_reward(), discount = np.asarray(100, dtype=np.float32))

    def _to_observation(self):
        # All values here are in the game frame of reference. We do the rotation at the end.
        vel_length = self._player.pod.vel.length()
        vel_angle = math.acos(self._player.pod.vel.x / vel_length) if vel_length > EPSILON else 0.0

        check1 = self._world.checkpoints[self._player.pod.nextCheckId]
        pod_to_check1 = check1 - self._player.pod.pos
        dist_to_check1 = pod_to_check1.length()
        ang_to_check1 = math.acos(pod_to_check1.x / dist_to_check1)

        check2 = self._world.checkpoints[(self._player.pod.nextCheckId + 1) % len(self._world.checkpoints)]
        check1_to_check2 = check2 - check1
        dist_check1_to_check2 = check1_to_check2.length()
        ang_check1_to_check2 = math.acos(check1_to_check2.x / dist_check1_to_check2)

        # Re-orient so pod is at (0,0) angle 0.0
        return np.array([
            clean_angle(vel_angle - self._player.pod.angle),
            clean_angle(ang_to_check1 - self._player.pod.angle),
            clean_angle(ang_check1_to_check2 - ang_to_check1 - self._player.pod.angle),
            vel_length,
            dist_to_check1,
            dist_check1_to_check2
        ])

    def _get_reward(self) -> int:
        reward = Constants.world_x() * Constants.world_y()
        reward += self._player.pod.nextCheckId * 10000
        reward -= (self._board.checkpoints[self._player.pod.nextCheckId] -  self._player.pod.pos).square_length()
        return np.asarray(reward, dtype=np.float32)
