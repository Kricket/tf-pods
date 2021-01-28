from typing import Union, List

import tensorflow as tf
import numpy as np

from pod.ai.ai_utils import NUM_ACTIONS, action_to_output, state_to_vector, get_best_action, STATE_VECTOR_LEN
from pod.board import PodBoard
from pod.controller import Controller, PlayInput, PlayOutput
from pod.game import game_step
from pod.util import PodState


class DeepRewardController(Controller):
    """
    A Controller that uses a NN to try to predict what action will produce
    the highest reward.
    """
    def __init__(self, board: PodBoard, model = None):
        self.board = board

        if model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    32,
                    input_shape=(STATE_VECTOR_LEN,),
                    kernel_initializer="zeros",
                    activation="sigmoid",
                ),
                tf.keras.layers.Dense(
                    NUM_ACTIONS,
                    kernel_initializer="zeros",
                ),
            ])
        else:
            self.model = model

        self.model.compile(
            optimizer=tf.optimizers.Adam(0.001),
            metrics=['accuracy'],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        self.best_actions = list(0 for x in range(NUM_ACTIONS))

    def play(self, pi: PlayInput) -> PlayOutput:
        output = self.__get_output(pi)
        best_action = np.argmax(output)
        return action_to_output(best_action, pi.angle, pi.pos)

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

    def __get_best_reward_output(self, pod: PodState):
        """
        Get an output array with the action with the highest reward set, and the others at 0
        """
        best_action = get_best_action(self.board, pod)
        self.best_actions[best_action] += 1
        result = np.zeros((1, NUM_ACTIONS))
        result[0][best_action] = 1.0
        return result

    def train(self, pods: List[PodState], epochs: int = 10):
        states = np.array(list(
            state_to_vector(
                p.pos,
                p.vel,
                p.angle,
                self.board.get_check(p.nextCheckId),
                self.board.get_check(p.nextCheckId + 1)
            ) for p in pods))
        labels = np.array(list(get_best_action(self.board, pod) for pod in pods))
        return self.model.fit(states, labels, epochs=epochs, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=5, min_delta=0.001)
        ])

    def train_online(self, turns: int = 200):
        states = []
        targets = []
        pod = PodState()

        for t in range(turns):
            states.append(state_to_vector(
                pod.pos, pod.vel, pod.angle,
                self.board.get_check(pod.nextCheckId), self.board.get_check(pod.nextCheckId + 1)))
            targets.append(self.__get_best_reward_output(pod))

            real_output = self.__get_output(pod)
            action = np.argmax(real_output)
            game_step(self.board, pod, action_to_output(action, pod.angle, pod.pos), pod)

        print("Best actions: {}".format(self.best_actions))
        return self.model.train_on_batch(np.array(states), np.array(targets))