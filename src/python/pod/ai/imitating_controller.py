import math
from random import random
from time import perf_counter
from typing import Tuple, List

import tensorflow as tf

import numpy as np
from async_util import run_multiproc
from pod.ai.action_discretizer import DiscreteActionController
from pod.ai.misc_controllers import DeepController
from pod.ai.replay_buffer import ReplayBuffer
from pod.ai.vectorizer import Vectorizer, V6
from pod.constants import Constants
from pod.util import PodState
from vec2 import UNIT


def _train_async(
        pid,
        num_episodes: int,
        init_states: List[PodState],
        target: DiscreteActionController,
        vectorizer: Vectorizer,
        max_turns: int,
) -> List[Tuple]:
    er = _EpisodeRunner(target, vectorizer)

    if init_states is None:
        init_states = [er.gen_initial_state() for i in range(num_episodes)]

    log_steps = math.floor(len(init_states) / 20)
    if log_steps < 1: log_steps = math.ceil(len(init_states) / 5)

    for (ep, pod) in enumerate(init_states):
        if ep % log_steps == 0: print("Proc {}: episode {} / {}".format(pid, ep, len(init_states)))
        er.run_episode(max_turns, pod.clone())

    return er.buffer

class _EpisodeRunner:
    def __init__(
            self,
            target: DiscreteActionController,
            vectorizer: Vectorizer
    ):
        self.target = target
        self.vectorizer = vectorizer
        self.buffer = []

    def run_episode(self, max_turns: int, pod: PodState):
        """
        Perform an episode: play the given pod a few turns and record what actions
        the target Controller takes
        """
        self.target.reset()

        for _ in range(max_turns):
            action = self.target.get_action(pod)

            # We want to save the action at the current state
            self.buffer.append((self.vectorizer.to_vector(self.target.board, pod), action))

            # Then, we use that action to step forward
            play = self.target.ad.action_to_output(action, pod.angle, pod.pos)
            self.target.board.step(pod, play, pod)

    def gen_initial_state(self) -> PodState:
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
        # The buffer stores (state, action) pairs
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

    def __map_labels(self, pods: List[PodState], num_episodes: int, n_proc: int, max_turns: int):
        start = perf_counter()

        if n_proc == 1:
            self.buffer.buffer += _train_async(
                1,
                num_episodes,
                pods,
                self.target,
                self.vectorizer,
                max_turns)
        else:
            if pods is None:
                split_pods = [None for _ in range(n_proc)]
            else:
                split_pods = np.array_split(pods, n_proc)

            results = run_multiproc(n_proc, _train_async, [(
                i,
                math.ceil(num_episodes / n_proc),
                split_pods[i],
                self.target,
                self.vectorizer,
                max_turns
            ) for i in range(n_proc)])
            for r in results: self.buffer.buffer += r

        end = perf_counter()
        print("Labels generated in %.3f seconds" % (end - start))

    def train_from_states(
            self,
            pods: List[PodState],
            max_turns: int,
            fit_epochs: int,
            n_proc: int = 1
    ):
        self.__map_labels(pods, 0, n_proc, max_turns)
        return self.learn(fit_epochs)

    def train_by_playing(
            self,
            num_episodes: int,
            max_turns: int,
            fit_epochs: int,
            n_proc: int = 1
    ):
        self.__map_labels(None, num_episodes, n_proc, max_turns)
        return self.learn(fit_epochs)

    def learn(self, epochs: int):
        print("Training for {} epochs with {} data points".format(epochs, len(self.buffer.buffer)))
        inputs, targets = zip(*self.buffer.buffer)

        history = self.model.fit(
            np.array(inputs),
            np.array(targets),
            epochs=epochs
        )

        return history.history['accuracy']
