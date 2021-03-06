import math
import random
from typing import List, Tuple

import tensorflow as tf

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer, DiscreteActionController
from pod.ai.replay_buffer import ReplayBuffer
from pod.ai.rewards import RewardFunc
from pod.ai.vectorizer import Vectorizer, V6
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState
from vec2 import UNIT


class DeepQController(DiscreteActionController):
    def __init__(self,
                 board: PodBoard,
                 reward_func: RewardFunc,
                 discretizer: ActionDiscretizer = ActionDiscretizer(),
                 vectorizer: Vectorizer = V6()):
        super().__init__(board, discretizer)

        self.reward_func = reward_func
        self.replay = ReplayBuffer()
        self.vectorizer = vectorizer

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                48,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                32,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                32,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                self.ad.num_actions,
                activation='linear',
            ),
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(0.001),
            metrics=['accuracy'],
            loss=tf.keras.losses.MeanSquaredError()
        )

    def get_action(self, pod: PodState) -> int:
        return self.__get_best_action(pod)

    def __get_output(self, pod: PodState = None, state_vec = None):
        """
        Get the output of the model for the given input state
        """
        if state_vec is None:
            state_vec = self.vectorizer.to_vector(self.board, pod)
        return self.model(tf.constant([state_vec]))

    def __get_best_action(self, pod: PodState = None, state_vec = None) -> int:
        """
        Get the action that produces the highest Q-value
        """
        output = self.__get_output(pod, state_vec)
        argmax = np.argmax(output)
#        print("Action {} because output {}".format(argmax, output))
        return argmax

    def __get_initial_pod(self) -> Tuple[PodState, List[float]]:
        # The pod starts in a random position at a random distance from the check,
        # pointing in a random direction
        pos_offset = UNIT.rotate(random.random() * 2 * math.pi) * \
                     Constants.check_radius() * (15 * random.random() + 1)
        pod = PodState(
            pos=self.board.get_check(0) + pos_offset,
            angle=2 * math.pi * random.random() - math.pi
        )
        state = self.vectorizer.to_vector(self.board, pod)
        return pod, state

    def __choose_action(self, state, prob_rand: float) -> int:
        """
        Choose an action for training: either random, or the one with the best Q-value
        """
        if random.random() < prob_rand:
            return math.floor(random.random() * self.ad.num_actions)
        else:
            return self.__get_best_action(state_vec=state)

    def __step(self, pod: PodState, action: int) -> Tuple[PodState, List[float], float]:
        """
        Get the result of the given pod taking the given action
        """
        next_pod = self.board.step(pod, self.ad.action_to_output(action, pod.angle, pod.pos), PodState())
        reward = self.reward_func(self.board, next_pod)

        return next_pod, self.vectorizer.to_vector(self.board, next_pod), reward

    def __run_episode(self, prob_rand_action: float, max_turns: int = 50):
        total_reward = 0
        pod, current_state = self.__get_initial_pod()
        while True:
            action = self.__choose_action(current_state, prob_rand_action)
            next_pod, next_state, reward = self.__step(pod, action)

#            self.replay.add((current_state, action, reward, next_state))
            self.replay.add((pod, current_state))

            total_reward += reward

            # Done when we get to the second checkpoint, or after too much time has passed
            pod = next_pod
            current_state = next_state
            if pod.nextCheckId > 1 or pod.turns > max_turns:
                break

        return total_reward

    def __get_q_update(self, next_state, reward: float, future_discount: float):
        return reward + future_discount * np.max(self.__get_output(state_vec=next_state))

    def __get_full_target_output(self, future_discount: float, pod: PodState) -> List[List[float]]:
        """
        For the given pod, get the target Q-values for every possible action.
        """
        target_q_values = []
        for action in range(self.ad.num_actions):
            next_pod, next_state, reward = self.__step(pod, action)
            target_q_values.append(self.__get_q_update(next_state, reward, future_discount))
#            max_next_q = np.max(self.__get_output(state_vec=next_state))
#            target_q_values.append(reward + future_discount * max_next_q)

        return [target_q_values]

    def __get_single_target_output(self,
                                   state,
                                   action: int,
                                   reward: float,
                                   next_state,
                                   future_discount: float
                                   ) -> List[List[float]]:
        """
        Get the target output, with a single value updated, for the given conditions
        """
        q_vec = self.__get_output(state_vec=state).numpy()
        q_vec[0][action] = self.__get_q_update(next_state, reward, future_discount)
            #reward + future_discount * np.max(self.__get_output(state_vec=next_state))
        return q_vec

    def __learn(self, future_discount: float):
#        inputs = [state for state, action, reward, next_state in self.replay.buffer]
        inputs = [state for pod, state in self.replay.buffer]

        print("Generating targets for {} states...".format(len(inputs)))
        targets = [
            self.__get_full_target_output(future_discount, pod)
            for pod, state in self.replay.buffer
#            self.__get_single_target_output(state, action, reward, next_state, future_discount)
#            for state, action, reward, next_state in self.replay.buffer
        ]

        history = self.model.fit(
            np.array(inputs),
            np.array(targets),
            epochs=5
        )

        return history.history['accuracy']

    def train(self,
              num_episodes: int = 300,
              exploration_episodes: int = 50,
              epsilon_decay: float = 0.95,
              max_turns: int = 40,
              future_discount: float = 0.7
              ):
        reward_per_ep = []
        accuracy = []
        epsilon = 1.0
        for episode in range(num_episodes):
            reward_per_ep.append(self.__run_episode(epsilon, max_turns))
            if episode > exploration_episodes:
                epsilon *= epsilon_decay
                accuracy += self.__learn(future_discount)

            print("Episode {}/{} epsilon {} total reward {}".format(
                episode, num_episodes, epsilon, reward_per_ep[-1]))

        return reward_per_ep, accuracy

    def train_from_examples(self,
                            pods: List[PodState],
                            num_episodes: int = 300,
                            exploration_episodes: int = 50,
                            epsilon_decay: float = 0.95,
                            future_discount: float = 0.7,
                            ):
        print("Building states for each pod...")
        pod_states = [self.vectorizer.to_vector(self.board, pod) for pod in pods]

        self.replay.capacity = len(pods)

        accuracy = []
        epsilon = 1.0
        for episode in range(num_episodes):
            print("Starting episode {}/{}  epsilon = {}".format(episode, num_episodes, epsilon))
            self.replay.clear()
            for pod, current_state in zip(pods, pod_states):
                action = self.__choose_action(current_state, epsilon)
                next_pod, next_state, reward = self.__step(pod, action)
                self.replay.add((pod, current_state))
#                self.replay.add(current_state, action, reward, next_state)

            accuracy += self.__learn(future_discount)

            if episode > exploration_episodes:
                epsilon *= epsilon_decay

        return accuracy
