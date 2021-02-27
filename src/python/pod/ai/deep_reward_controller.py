from typing import List, Callable

import tensorflow as tf

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer
from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput
from pod.util import PodState

# Size of the vector returned by state_to_vector
STATE_VECTOR_LEN = 4

def state_to_vector(pod: PodState, board: PodBoard) -> List[float]:
    """
    Convert a PodState to an input vector that can be used for the neural network
    """
    # Velocity is already relative to the pod, so it just needs to be rotated
    vel = pod.vel.rotate(-pod.angle) / Constants.max_vel()

    check1 = (board.get_check(pod.nextCheckId) - pod.pos).rotate(-pod.angle) / MAX_DIST

    return [vel.x, vel.y, check1.x, check1.y]


class DeepRewardController(Controller):
    """
    A Controller that uses a NN to try to predict what action will produce
    the highest reward.
    """
    def __init__(self,
                 board: PodBoard,
                 reward_func: Callable[[PodBoard, PodState, PodState], float],
                 model=None,
                 discretizer: ActionDiscretizer = ActionDiscretizer()):
        super().__init__(board)

        self.reward_func = reward_func
        self.ad = discretizer

        if model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    32,
                    input_shape=(STATE_VECTOR_LEN,),
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

        self.best_actions = list(0 for _ in range(self.ad.num_actions))

    def play(self, pod: PodState) -> PlayOutput:
        output = self.__get_output(pod)
        best_action = np.argmax(output)
        return self.ad.action_to_output(best_action, pod.angle, pod.pos)

    def __get_output(self, pod: PodState):
        """
        Get the output of the model for the given input state
        """
        input_vec = state_to_vector(pod, self.board)
        return self.model(tf.constant([input_vec]))

    def __get_best_reward_output(self, pod: PodState):
        """
        Get an output array with the action with the highest reward set, and the others at 0
        """
        best_action = self.ad.get_best_action(self.board, pod, self.reward_func)
        self.best_actions[best_action] += 1
        result = np.zeros((1, self.ad.num_actions))
        result[0][best_action] = 1.0
        return result

    def train(self, pods: List[PodState], epochs: int = 10):
        states = np.array(list(state_to_vector(p, self.board) for p in pods))
        labels = np.array(list(self.ad.get_best_action(self.board, pod, self.reward_func) for pod in pods))
        return self.model.fit(states, labels, epochs=epochs, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=5, min_delta=0.001)
        ])
