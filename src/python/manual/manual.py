import tensorflow as tf

from pod.board import PodBoard
from pod.controller import Controller, PlayOutput, PlayInput
from pod.util import PodState
from qnetwork.qpod_env import THRUST_VALUES, ANGLE_VALUES, state_to_vector, action_to_output


class DQController(Controller):
    def __init__(self, board: PodBoard):
        self.__board = board

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128,
                input_shape=(6,),
                kernel_initializer="random_normal",
                bias_initializer="random_normal"
            ),
            tf.keras.layers.Dense(
                THRUST_VALUES * ANGLE_VALUES,
                kernel_initializer="random_normal",
                bias_initializer="random_normal"
            )
        ])

        # self.model.compile(
        #     optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0),
        #     loss=tf.keras.losses.mean_squared_error()
        # )

    def play(self, pi: PlayInput) -> PlayOutput:
        input_vec = state_to_vector(
            pi.pos,
            pi.vel,
            pi.angle,
            self.__board.get_check(pi.nextCheckId),
            self.__board.get_check(pi.nextCheckId + 1))
        q_values = self.model(tf.constant([input_vec]))
        best_idx = tf.keras.backend.get_value(tf.argmax(q_values, 1))
        return action_to_output(best_idx, pi.angle, pi.pos)


class Trainer:
    def __init__(self, board: PodBoard):
        self.pod = PodState()
        self.board = board
