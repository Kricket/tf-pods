import random
from typing import List, Union

import tensorflow as tf

import numpy as np
from pod.ai.ai_utils import THRUST_VALUES, ANGLE_VALUES, state_to_vector, action_to_output, reward
from pod.board import PodBoard
from pod.controller import Controller, PlayOutput, PlayInput
from pod.game import game_step
from pod.util import PodState


class DQController(Controller):
    def __init__(self, board: PodBoard):
        self.board = board

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                64,
                input_shape=(6,),
                kernel_initializer="zeros",
                activation="sigmoid"
            ),
            tf.keras.layers.Dense(
                THRUST_VALUES * ANGLE_VALUES,
                kernel_initializer="zeros",
                activation="sigmoid"
            )
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.mean_squared_error
        )

    def play(self, pi: PlayInput) -> PlayOutput:
        #q_values = self.__get_output(pi)
        #best_idx = np.argmax(q_values.numpy())
        q_values = self.__get_best_reward_output(pi.as_pod())
        best_idx = np.argmax(q_values)

        print("Best idx of {} is {}".format(q_values, best_idx))
        return action_to_output(best_idx, pi.angle, pi.pos)

    def __get_output(self, pi: Union[PlayInput, PodState]):
        """
        Get the output of the model for the given input state
        """
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
        q_values = self.__get_output(pod)
        best_q_idx = \
            tf.keras.backend.get_value(tf.argmax(q_values, 1))[0] if random.random() < epsilon \
            else int(random.random() * THRUST_VALUES * ANGLE_VALUES)

        next_state = PodState()
        game_step(self.board, pod, action_to_output(best_q_idx, pod.angle, pod.pos), next_state)

        next_q_values = self.__get_output(pod)
        best_next_q = tf.keras.backend.get_value(tf.math.reduce_max(next_q_values))

        numpy_q = q_values.numpy()
        numpy_q[0][best_q_idx] = reward(next_state, self.board) + discount * best_next_q
        return numpy_q, best_q_idx

    def __get_best_reward_output(self, pod: PodState):
        """
        Get an output array with the action with the highest reward set, and the others at 0
        """
        num_actions = THRUST_VALUES * ANGLE_VALUES

        best_action = 0
        best_reward = -999

        for action in range(0, num_actions):
            next_state = PodState()
            game_step(self.board, pod, action_to_output(action, pod.angle, pod.pos), next_state)
            r = reward(next_state, self.board)
            if r > best_reward:
                best_reward = r
                best_action = action

        result = np.zeros((1, num_actions))
        result[0][best_action] = 1
        print("State {} best action {}".format(pod, action_to_output(best_action, pod.angle, pod.pos)))
        print("    {} --> {}".format(best_action, result))
        return result

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
                self.__get_best_reward_output(pod) \
                    if use_best else
                    self.__get_target_q_values(pod, discount, epsilon)[0]
                for pod in pods[start:start + batch_size]))
            loss = self.model.train_on_batch(batch_states, batch_qs)
            print("Training on batch {} - {}: loss = {}".format(start, start + batch_size, loss))
            start += batch_size

    def train_rewards_online(self, iterations: int, batch_size: int = 30):
        """
        Train by starting at a given point and gathering data as we go. Aim for whatever gives the highest reward.
        """
        pod = PodState()
        total_loss = 0
        for i in range(0, iterations):
            states = []
            targets = []
            for b in range(0, batch_size):
                state = state_to_vector(
                    pod.pos,
                    pod.vel,
                    pod.angle,
                    self.board.get_check(pod.nextCheckId),
                    self.board.get_check(pod.nextCheckId + 1))
                states.append(state)
                targets.append(self.__get_best_reward_output(pod))
                action = np.argmax(self.__get_output(pod).numpy())
                game_step(self.board, pod, action_to_output(action, pod.angle, pod.pos), pod)
            total_loss += self.model.train_on_batch(np.array(states), np.array(targets))
        print("Average loss: {}".format(total_loss / iterations))

    def train_rewards(self, pods: List[PodState], batch_size: int = 30):
        """
        Train the network to simply produce the action with the best reward
        """
        states = np.array(list(
            state_to_vector(
                p.pos,
                p.vel,
                p.angle,
                self.board.get_check(p.nextCheckId),
                self.board.get_check(p.nextCheckId + 1)
            ) for p in pods))

        start = 0
        losses = []
        while start < len(states):
            batch_states = states[start:start + batch_size]
            batch_rewards = np.array(list(
                self.__get_best_reward_output(pod)
                for pod in pods[start:start + batch_size]))
            loss = self.model.train_on_batch(batch_states, batch_rewards)
            losses.append(loss)
            start += batch_size

        print("Average loss: {}".format(np.average(losses)))