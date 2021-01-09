from typing import Any

import tensorflow as tf

import numpy as np
from pod.ai.ai_utils import action_to_output, MAX_ACTION, state_to_vector, reward
from pod.board import PodBoard
from pod.controller import Controller, PlayOutput, PlayInput
from pod.game import Player
from pod.util import PodState
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class QPodController(Controller):
    def __init__(self):
        self.play_output = PlayOutput()

    def set_play(self, action, pod: PodState):
        """
        Convert the action to a PlayOutput
        """
        action_to_output(action.item(), pod.angle, pod.pos, self.play_output)

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
        return state_to_vector(
            self._player.pod.pos,
            self._player.pod.vel,
            self._player.pod.angle,
            self._board.get_check(self._player.pod.nextCheckId),
            self._board.get_check(self._player.pod.nextCheckId + 1)
        )

    def _get_reward(self) -> int:
        return np.asarray(reward(self._player.pod, self._board), dtype=np.float32)
