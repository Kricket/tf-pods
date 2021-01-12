import math
from typing import List, Union

import random

import numpy as np
import tensorflow as tf

from pod.ai.ai_utils import THRUST_VALUES, ANGLE_VALUES, state_to_vector, action_to_output, reward, MAX_VEL
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput, PlayInput
from pod.game import game_step
from pod.util import PodState
from vec2 import UNIT


class DQController(Controller):
    def __init__(self, board: PodBoard):
        self.board = board

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                100,
                input_shape=(6,),
                kernel_initializer="zeros",
                activation="tanh"
            ),
            tf.keras.layers.Dense(
                100,
                kernel_initializer="zeros",
                activation="tanh"
            ),
            tf.keras.layers.Dense(
                THRUST_VALUES * ANGLE_VALUES,
                kernel_initializer="zeros"
            )
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.mean_squared_error
        )

    def play(self, pi: Union[PlayInput, PodState]) -> PlayOutput:
        q_values = self.__get_q_values(pi)
        best_idx = tf.keras.backend.get_value(tf.argmax(q_values, 1))[0]
        return action_to_output(best_idx, pi.angle, pi.pos)

    def __get_q_values(self, pi: Union[PlayInput, PodState]):
        input_vec = state_to_vector(
            pi.pos,
            pi.vel,
            pi.angle,
            self.board.get_check(pi.nextCheckId),
            self.board.get_check(pi.nextCheckId + 1))
        return self.model(tf.constant([input_vec]))

    def __get_target_q_values(self, pod: PodState, discount: float, epsilon: float):
        """
        Get the Q-value vector to update for the given state, i.e. Q(s, a) for all actions for the given pod state
        """
        q_values = self.__get_q_values(pod)
        best_q_idx = \
            tf.keras.backend.get_value(tf.argmax(q_values, 1))[0] if random.random() < epsilon \
            else int(random.random() * THRUST_VALUES * ANGLE_VALUES)

        next_state = PodState()
        game_step(self.board, pod, action_to_output(best_q_idx, pod.angle, pod.pos), next_state)

        next_q_values = self.__get_q_values(pod)
        best_next_q = tf.keras.backend.get_value(tf.math.reduce_max(next_q_values))

        numpy_q = q_values.numpy()
        numpy_q[0][best_q_idx] = reward(next_state, self.board) + discount * best_next_q
        return numpy_q

    def __get_target_highest_reward_values(self, pod: PodState):
        """
        Get an output array with all zeroes, but 1 in the place of the action with the highest reward
        """
        rewards = []
        for action in range(0, THRUST_VALUES * ANGLE_VALUES):
            next_state = PodState()
            game_step(self.board, pod, action_to_output(action, pod.angle, pod.pos), next_state)
            rewards.append(reward(next_state, self.board))
        return np.array([rewards])

    def gen_pods(
            self,
            # Number of points on a circle around the target checkpoint
            num_pod_pos: int,
            # Number of distances (radius of circle around the target checkpoint)
            num_pod_dist: int,
            # Number of angles in which the pod is facing (from 0 to 2pi)
            num_pod_angles: int,
            # Number of velocity lengths (from 0 to MAX_VEL)
            num_vel_len: int,
            # Number of velocity directions (from 0 to 2pi)
            num_vel_angles: int
    ) -> List[PodState]:
        pods = []

        for pod_pos_angle_deg in range(0, 360, int(360 / num_pod_pos)):
            pod_pos_dir = UNIT.rotate(math.radians(pod_pos_angle_deg))
            for pod_dist in range(Constants.check_radius() + 1, Constants.world_x(), int((Constants.world_x() - Constants.check_radius()) / num_pod_dist)):
                pos = pod_pos_dir * pod_dist + self.board.checkpoints[0]
                for pod_angle_deg in range(0, 360, int(360 / num_pod_angles)):
                    angle = math.radians(pod_angle_deg)
                    for vel_angle_deg in range(0, 360, int(360 / num_vel_angles)):
                        vel_dir = UNIT.rotate(math.radians(vel_angle_deg))
                        for vel_len in range(0, MAX_VEL, int(MAX_VEL / num_vel_len)):
                            pod = PodState(pos)
                            pod.vel = vel_dir * vel_len
                            pod.angle = angle
                            pods.append(pod)

        np.random.shuffle(pods)
        print("{} states generated".format(len(pods)))
        return pods

    def train(self, pods: List[PodState], batch_size: int = 100, discount: float = 0.1, epsilon: float = 0, use_best = False):
        # TODO: vary the check distances
        states = np.array(list(
            state_to_vector(
                p.pos,
                p.vel,
                p.angle,
                self.board.get_check(p.nextCheckId),
                self.board.get_check(p.nextCheckId + 1)
            ) for p in pods))

        start = 0
        while start < len(states):
            batch_states = states[start:start + batch_size]
            batch_qs = np.array(list(
                self.__get_target_highest_reward_values(pod) \
                    if use_best else
                    self.__get_target_q_values(pod, discount, epsilon)
                for pod in pods[start:start + batch_size]))
            loss = self.model.train_on_batch(batch_states, batch_qs)
            print("Training on batch {} - {}: loss = {}".format(start, start + batch_size, loss))
            start += batch_size
