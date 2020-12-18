from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Any

import tensorflow as tf
import numpy as np
from constants import Constants
from podutil import PodInfo, Controller, PlayInput, PlayOutput
from podworld import PodWorld

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from vec2 import ORIGIN

tf.compat.v1.enable_v2_behavior()


class AgentController(Controller):
    def __init__(self):
        self.play_output = PlayOutput()

    def set_play(self, action, pod: PodInfo):
        """
        Convert the action to a PlayInput
        """
        self.play_output.thrust = action['thrust']
        angle = action['angle'] + pod.angle
        rel_dir = ORIGIN.rotate(angle) * 100
        self.play_output.target = pod.pos + rel_dir

    def play(self, pi: PlayInput) -> PlayOutput:
        return self.play_output


class PodEnvironment(py_environment.PyEnvironment):

    def __init__(self, world: PodWorld):
        super().__init__()

        angle_spec = array_spec.BoundedArraySpec(
            (),
            np.float,
            minimum=-Constants.max_turn(),
            maximum=Constants.max_turn())
        thrust_spec = array_spec.BoundedArraySpec(
            (),
            np.int32,
            minimum=0,
            maximum=100)
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
            maximum=Constants.world_x() * 2)

        self._observation_spec = {
            'angles': angles_spec,
            'distances': dist_spec
        }

        self._world = world
        self._controller = AgentController()
        world.add_player(self._controller)
        self._player_idx = len(world.players) - 1
        self._initial_state = world.serialize()
        self._episode_ended = False

    def _get_pod(self) -> PodInfo:
        return self._world.players[self._player_idx].pod

    def get_state(self) -> Any:
        return self._world.serialize()

    def set_state(self, state: Any) -> None:
        self._world.deserialize(state)

    def get_info(self) -> Any:
        raise NotImplementedError("What is this?")

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.set_state(self._initial_state)
        self._episode_ended = False
        # TODO: this should be an observation
        return ts.restart(self._to_observation())

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        pod = self._get_pod()
        if pod.nextCheckId > 1 or self._world.turns > 100:
            # That's enough for training...
            self._episode_ended = True
        else:
            # Play the given action
            self._controller.set_play(action, pod)
            self._world.step()

        if self._episode_ended:
            return ts.termination(self._to_observation(), self._get_reward())
        else:
            return ts.transition(self._to_observation(), reward = self._get_reward(), discount = 100)

    def _to_observation(self):
        pod = self._get_pod()

        # All values here are in the game frame of reference. We do the rotation at the end.
        vel_length = pod.vel.length()
        vel_angle = math.acos(pod.vel.x / vel_length)

        check1 = self._world.checkpoints[pod.nextCheckId]
        pod_to_check1 = check1 - pod.pos
        dist_to_check1 = pod_to_check1.length()
        ang_to_check1 = math.acos(pod_to_check1.x / dist_to_check1)

        check2 = self._world.checkpoints[(pod.nextCheckId + 1) % len(self._world.checkpoints)]
        check1_to_check2 = check2 - check1
        dist_check1_to_check2 = check1_to_check2.length()
        ang_check1_to_check2 = math.acos(check1_to_check2.x / dist_check1_to_check2)

        # Re-orient so pod is at (0,0) angle 0.0
        return {
            'angles': [
                vel_angle - pod.angle,
                ang_to_check1 - pod.angle,
                ang_check1_to_check2 - ang_to_check1 - pod.angle
            ],
            'distances': [
                vel_length,
                dist_to_check1,
                dist_check1_to_check2,
            ]
        }

    def _get_reward(self) -> int:
        pod = self._get_pod()
        reward = Constants.world_x() * Constants.world_y()
        reward += pod.nextCheckId * 10000
        reward -= (self._world.checkpoints[pod.nextCheckId] -  pod.pos).square_length()
        return reward
