from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Any

import tensorflow as tf
import numpy as np
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput, PlayInput
from pod.game import Player
from pod.util import PodState, clean_angle

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from vec2 import ORIGIN, EPSILON

tf.compat.v1.enable_v2_behavior()


""" Amount by which we allow the agent to go above or below the real thrust bounds """
THRUST_PADDING = 10


class AgentController(Controller):
    def __init__(self):
        self.play_output = PlayOutput()

    def set_play(self, action, pod: PodState):
        """
        Convert the action to a PlayInput
        """
        self.play_output.thrust = action['thrust'] - THRUST_PADDING
        angle = action['angle'] + pod.angle
        rel_dir = ORIGIN.rotate(angle) * 1000
        self.play_output.target = pod.pos + rel_dir

    def play(self, pi: PlayInput) -> PlayOutput:
        return self.play_output


class PodEnvironment(py_environment.PyEnvironment):

    def __init__(self, board: PodBoard):
        super().__init__()

        # Allow the agent to go beyond the bounds - due to the nature of
        # the rounding functions, it's unlikely the agent will ever give
        # us the actual min or max
        scaled_max_turn = Constants.max_turn() * 1.1
        scaled_max_thrust = Constants.max_thrust() + 2 * THRUST_PADDING
        angle_spec = array_spec.BoundedArraySpec(
            (),
            np.float,
            minimum=-scaled_max_turn,
            maximum=scaled_max_turn)
        thrust_spec = array_spec.BoundedArraySpec(
            (),
            np.int32,
            minimum=0,
            maximum=scaled_max_thrust)
        self._action_spec = {
            'angle': angle_spec,
            'thrust': thrust_spec
        }

        angles_spec = array_spec.BoundedArraySpec(
            (3,),
            np.float,
            minimum=-math.pi,
            maximum=math.pi)
        dist_spec = array_spec.BoundedArraySpec(
            (3,),
            np.float,
            minimum=0,
            maximum=Constants.world_x() * 10)

        self._observation_spec = {
            'angles': angles_spec,
            'distances': dist_spec
        }

        self._time_step_spec = ts.TimeStep(
            step_type=array_spec.ArraySpec(shape=(), dtype=np.int32, name='step_type'),
            reward=array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward'),
            discount=array_spec.ArraySpec(shape=(), dtype=np.float32, name='discount'),
            observation=self._observation_spec
        )

        self._board = board
        self._player = Player(AgentController())
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

        if self._player.pod.nextCheckId > 1 or self._player.pod.turns > 100:
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

        check1 = self._board.checkpoints[self._player.pod.nextCheckId]
        pod_to_check1 = check1 - self._player.pod.pos
        dist_to_check1 = pod_to_check1.length()
        ang_to_check1 = math.acos(pod_to_check1.x / dist_to_check1)

        check2 = self._world.checkpoints[(self._player.pod.nextCheckId + 1) % len(self._world.checkpoints)]
        check1_to_check2 = check2 - check1
        dist_check1_to_check2 = check1_to_check2.length()
        ang_check1_to_check2 = math.acos(check1_to_check2.x / dist_check1_to_check2)

        # Re-orient so pod is at (0,0) angle 0.0
        return {
            'angles': np.array([
                clean_angle(vel_angle - self._player.pod.angle),
                clean_angle(ang_to_check1 - self._player.pod.angle),
                clean_angle(ang_check1_to_check2 - ang_to_check1 - self._player.pod.angle)
            ]),
            'distances': np.array([
                vel_length,
                dist_to_check1,
                dist_check1_to_check2,
            ])
        }

    def _get_reward(self) -> int:
        reward = Constants.world_x() * Constants.world_y()
        reward += self._player.pod.nextCheckId * 10000
        reward -= (self._world.checkpoints[self._player.pod.nextCheckId] -  self._player.pod.pos).square_length()
        return np.asarray(reward, dtype=np.float32)
