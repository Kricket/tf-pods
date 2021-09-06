import math
import random
from statistics import mean
from typing import List, Tuple

import tensorflow as tf
from tensorflow.python.keras import Model

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer, DiscreteActionController
from pod.ai.replay_buffer import ReplayBuffer
from pod.ai.rewards import RewardFunc, check_and_speed_reward
from pod.ai.vectorizer import Vectorizer, V6
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState
from vec2 import UNIT


class DeepQController(DiscreteActionController):
    def __init__(self,
                 board: PodBoard,
                 reward_func: RewardFunc = check_and_speed_reward,
                 discretizer: ActionDiscretizer = ActionDiscretizer(),
                 vectorizer: Vectorizer = V6()):
        super().__init__(board, discretizer)

        self.reward_func = reward_func
        self.replay = ReplayBuffer()
        self.vectorizer = vectorizer

        self.policy_net = self.__build_model()
        self.target_net = self.__build_model()
        self.optimizer = tf.optimizers.Adam(0.001)

    def __build_model(self) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                48,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(),
            ),
            tf.keras.layers.Dense(
                32,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(),
            ),
            tf.keras.layers.Dense(
                32,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(),
            ),
            tf.keras.layers.Dense(
                self.ad.num_actions,
                activation='linear',
            ),
        ])

    def get_action(self, pod: PodState) -> int:
        return self.__get_best_action(self.target_net, pod)

    def __get_output(self, model: Model, pod: PodState = None, state_vec = None):
        """
        Get the output of the model for the given input state
        """
        if state_vec is None:
            state_vec = self.vectorizer.to_vector(self.board, pod)
        return model(tf.constant([state_vec]))

    def __get_best_action(self, model: Model, pod: PodState = None, state_vec = None) -> int:
        """
        Get the action that produces the highest Q-value
        """
        output = self.__get_output(model, pod, state_vec)
        argmax = np.argmax(output)
        return argmax

    def __get_initial_pod(self, dist_increment: int, incr: int):
        # Position is (radius + increment) distance from check
        pos_offset = UNIT.rotate(random.random() * 2 * math.pi) * \
                     (Constants.check_radius() + max(dist_increment * incr, 5))
        return PodState(
            pos=self.board.checkpoints[0] + pos_offset,
            angle=2 * math.pi * random.random() - math.pi
        )

    def __choose_action(self, state: List[float], prob_rand_action: float):
        if random.random() < prob_rand_action:
            return math.floor(random.random() * self.ad.num_actions)
        else:
            return np.argmax(self.__get_best_action(self.policy_net, state_vec=state))

    def __step(self, pod: PodState, action: int):
        play = self.ad.action_to_output(action, pod.angle, pod.pos)
        next_pod = self.board.step(pod, play)
        reward = self.reward_func(self.board, next_pod)
        return next_pod, reward

    def __update_policy(self, batch_size: int, gamma: float):
        if len(self.replay.buffer) < batch_size: return 0

        batch = (*zip(*(self.replay.sample(batch_size))),)

        states = np.asarray(batch[0])
        actions = np.asarray(batch[1])
        next_states = np.asarray(batch[2])
        rewards = np.asarray(batch[3])
        dones = np.asarray(batch[4])

        q_s_a_prime = np.max(self.target_net(np.atleast_2d(next_states).astype('float32')), axis = 1)
        q_s_a_target = np.where(
            dones,
            rewards,
            # If not done, the reward will be 0
            #rewards +
            gamma * q_s_a_prime)
        q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')

        with tf.GradientTape() as tape:
            q_s_a = tf.math.reduce_sum(
                self.policy_net(
                    np.atleast_2d(states).astype('float32')
                ) * tf.one_hot(actions, self.ad.num_actions),
                axis=1)
            loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

        variables = self.policy_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()

    def __copy_weights(self):
        for policy, target in zip(self.policy_net.trainable_variables, self.target_net.trainable_variables):
            target.assign(policy.numpy())

    def train_progressively(
            self,
            prob_rand_action: float,
            dist_increment: int,
            ep_per_dist: int,
            num_incr: int,
            batch_size: int = 50,
            learning_rate: float = 0.8
    ) -> Tuple[List[float], List[float]]:
        max_reward_per_ep = []
        avg_losses = []
        total_steps = 0

        for incr in range(num_incr):
            max_turns = 5 * (incr + 1)
            for ep_inc in range(ep_per_dist):
                pod = self.__get_initial_pod(dist_increment, incr)
                pod_state = self.vectorizer.to_vector(self.board, pod)
                ep_reward = 0
                losses = []

                while pod.turns < max_turns and pod.nextCheckId == 0:
                    action = self.__choose_action(pod_state, prob_rand_action)
                    next_pod, reward = self.__step(pod, action)
                    next_state = self.vectorizer.to_vector(self.board, pod)

                    self.replay.add((pod_state, action, next_state, reward, pod.nextCheckId > 0))
                    ep_reward += reward

                    losses.append(self.__update_policy(batch_size, learning_rate))

                    pod = next_pod
                    pod_state = next_state
                    total_steps += 1

                    if total_steps % 100 == 0:
                        self.__copy_weights()

                if len(losses) > 0:
                    avg_losses.append(mean(losses))
                max_reward_per_ep.append(ep_reward)

        return max_reward_per_ep, avg_losses
