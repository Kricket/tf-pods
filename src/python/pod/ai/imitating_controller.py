import math
from random import random

import tensorflow as tf
import numpy as np

from pod.ai.action_discretizer import DiscreteActionController
from pod.ai.misc_controllers import DeepController
from pod.ai.replay_buffer import ReplayBuffer
from pod.ai.vectorizer import Vectorizer, V6
from pod.constants import Constants
from pod.util import PodState
from vec2 import UNIT


class ImitatingController(DeepController):
    """
    Controller that trains a NN to create the same output as another Controller
    """
    def __init__(self, target: DiscreteActionController, vectorizer: Vectorizer = V6()):
        super().__init__(target.board, vectorizer, target.ad)
        self.target = target
        self.buffer = ReplayBuffer(50000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                48,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                32,
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                24,
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                self.ad.num_actions,
            ),
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(0.002),
            metrics=['accuracy'],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

    def train(self, num_episodes: int, max_turns: int, fit_epochs: int):
        for episode in range(num_episodes):
            if episode % 20 == 0: print("--------------- Episode {}".format(episode))
            pod = self._get_initial_state()
            self._run_episode(pod, max_turns)

        return self._learn(fit_epochs)

    def _get_initial_state(self) -> PodState:
        """
        Generate a state at which to start a training episode
        """
        # The pod starts in a random position at a random distance from the check,
        # pointing in a random direction
        pos_offset = UNIT.rotate(random() * 2 * math.pi) * \
                     Constants.check_radius() * (15 * random() + 1)
        return PodState(
            pos=self.board.get_check(0) + pos_offset,
            angle=2 * math.pi * random() - math.pi
        )

    def _run_episode(self, pod: PodState, max_turns: int):
        """
        Perform an episode: play the given pod a few turns and record what actions
        the target Controller takes
        """
        self.target.reset()

        for _ in range(max_turns):
            action = self.target.get_action(pod)

            # We want to save the action at the current state
            self.buffer.add((self.vectorizer.to_vector(self.board, pod), action))

            # Then, we use that action to step forward
            play = self.ad.action_to_output(action, pod.angle, pod.pos)
            self.board.step(pod, play, pod)

    def _learn(self, epochs: int):
        inputs, targets = zip(*self.buffer.buffer)

        history = self.model.fit(
            np.array(inputs),
            np.array(targets),
            epochs=epochs
        )

        return history.history['accuracy']
