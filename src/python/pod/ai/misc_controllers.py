import math
from random import random

import tensorflow as tf
import numpy as np

from pod.ai.action_discretizer import ActionDiscretizer
from pod.ai.vectorizer import Vectorizer
from pod.board import PlayOutput, PodBoard
from pod.constants import Constants
from pod.controller import Controller
from pod.util import PodState
from vec2 import Vec2


class RandomController(Controller):
    """
    Plays a random move every turn
    """
    def play(self, pod: PodState) -> PlayOutput:
        return PlayOutput(
            Vec2(random() * Constants.world_x(), random() * Constants.world_y()),
            math.ceil(random() * Constants.max_thrust())
        )


class DeepController(Controller):
    """
    Abstract base class for a Controller that uses a NN to make plays
    """
    def __init__(self, board: PodBoard,
                 vectorizer: Vectorizer,
                 discretizer: ActionDiscretizer = ActionDiscretizer()):
        super().__init__(board)
        self.vectorizer = vectorizer
        self.ad = discretizer
        self.model = None

    def play(self, pod: PodState) -> PlayOutput:
        output = self._get_output(pod)
        best_action = np.argmax(output)
        return self.ad.action_to_output(best_action, pod.angle, pod.pos)

    def _get_output(self, pod: PodState):
        """
        Get the output of the model for the given input state
        """
        input_vec = self.vectorizer.to_vector(self.board, pod)
        return self.model(tf.constant([input_vec]))
