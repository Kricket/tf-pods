from multiprocessing import Pool
from time import perf_counter
import math
from random import random
from typing import Tuple, List

import tensorflow as tf

import numpy as np
from pod.ai.action_discretizer import DiscreteActionController
from pod.ai.misc_controllers import DeepController
from pod.ai.replay_buffer import ReplayBuffer
from pod.ai.vectorizer import Vectorizer, V6
from pod.constants import Constants
from pod.util import PodState
from vec2 import UNIT


def _train_async(pid,
                 num_episodes: int,
                 target: DiscreteActionController,
                 vectorizer: Vectorizer,
                 max_turns: int) -> List[Tuple]:
    er = _EpisodeRunner(target, vectorizer)

    for ep in range(num_episodes):
        if ep % 10 == 0: print("Proc {}: episode {}".format(pid, ep))
        er.run_episode(max_turns)

    return er.buffer

class _EpisodeRunner:
    def __init__(
            self,
            target: DiscreteActionController,
            vectorizer: Vectorizer,
    ):
        self.target = target
        self.vectorizer = vectorizer
        self.buffer = []

    def run_episode(self, max_turns: int):
        """
        Perform an episode: play the given pod a few turns and record what actions
        the target Controller takes
        """
        self.target.reset()
        pod = self._get_initial_state()

        for _ in range(max_turns):
            action = self.target.get_action(pod)

            # We want to save the action at the current state
            self.buffer.append((self.vectorizer.to_vector(self.target.board, pod), action))

            # Then, we use that action to step forward
            play = self.target.ad.action_to_output(action, pod.angle, pod.pos)
            self.target.board.step(pod, play, pod)

    def _get_initial_state(self) -> PodState:
        """
        Generate a state at which to start a training episode
        """
        # The pod starts in a random position at a random distance from the check,
        # pointing in a random direction
        pos_offset = UNIT.rotate(random() * 2 * math.pi) * \
                     Constants.check_radius() * (15 * random() + 1)

        return PodState(
            pos=self.target.board.get_check(0) + pos_offset,
            angle=2 * math.pi * random() - math.pi
        )


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
                64,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                input_shape=(self.vectorizer.vec_len(),),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                48,
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            ),
            tf.keras.layers.Dense(
                32,
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

    def train(self,
              num_episodes: int,
              max_turns: int,
              fit_epochs: int,
              n_proc: int = 1):
        start = perf_counter()

        if n_proc == 1:
            self.buffer.buffer += _train_async(1, num_episodes, self.target, self.vectorizer, max_turns)
        else:
            with Pool(n_proc) as p:
                results = p.starmap(_train_async, [(
                    i,
                    math.ceil(num_episodes / n_proc),
                    self.target,
                    self.vectorizer,
                    max_turns
                ) for i in range(n_proc)])

                for r in results: self.buffer.buffer += r
                p.close()
                p.join()

        end = perf_counter()
        print("Episodes generated in %.3f seconds" % (end - start))

        return self._learn(fit_epochs)

    def _learn(self, epochs: int):
        print("Training for {} epochs with {} data points".format(epochs, len(self.buffer.buffer)))
        inputs, targets = zip(*self.buffer.buffer)

        history = self.model.fit(
            np.array(inputs),
            np.array(targets),
            epochs=epochs
        )

        return history.history['accuracy']
