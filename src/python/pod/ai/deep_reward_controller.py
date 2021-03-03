from typing import List, Callable

import tensorflow as tf

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer
from pod.ai.misc_controllers import DeepController
from pod.ai.vectorizer import Vectorizer, V4
from pod.board import PodBoard
from pod.util import PodState


class DeepRewardController(DeepController):
    """
    A Controller that uses a NN to try to predict what action will produce
    the highest reward.
    """
    def __init__(self,
                 board: PodBoard,
                 reward_func: Callable[[PodBoard, PodState, PodState], float],
                 model=None,
                 discretizer: ActionDiscretizer = ActionDiscretizer(),
                 vectorizer: Vectorizer = V4()):
        super().__init__(board, vectorizer, discretizer)

        self.reward_func = reward_func

        if model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    32,
                    input_shape=(self.vectorizer.vec_len(),),
                    kernel_initializer="zeros",
                    activation="sigmoid",
                ),
                tf.keras.layers.Dense(
                    self.ad.num_actions,
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

    def __get_best_reward_output(self, pod: PodState):
        """
        Get an output array with the action with the highest reward set, and the others at 0
        """
        best_action = self.ad.get_best_action(self.board, pod, self.reward_func)
        result = np.zeros((1, self.ad.num_actions))
        result[0][best_action] = 1.0
        return result

    def train(self, pods: List[PodState], epochs: int = 10):
        states = np.array([self.vectorizer.to_vector(self.board, p) for p in pods])
        labels = np.array([self.ad.get_best_action(self.board, pod, self.reward_func) for pod in pods])
        return self.model.fit(states, labels, epochs=epochs, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=5, min_delta=0.001)
        ])
