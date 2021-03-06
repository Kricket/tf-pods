import math
from multiprocessing import Pool
from time import perf_counter
from typing import Callable, List, Tuple

import tensorflow as tf

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer
from pod.ai.misc_controllers import DeepController
from pod.ai.vectorizer import Vectorizer, V6
from pod.board import PodBoard
from pod.util import PodState


def _build_labels(ad: ActionDiscretizer,
                  board: PodBoard,
                  pods: List[Tuple[PodState, List[float]]],
                  r_func
                  ) -> List[float]:
    count = 0
    labels = []
    for t in pods:
        labels.append(ad.get_best_action(board, t[0], r_func))
        count += 1
        if count % 1000 == 0:
            print("{} labels done...".format(count))
    return labels


class DeepTreeController(DeepController):
    def __init__(self,
                 board: PodBoard,
                 reward_func: Callable[[PodBoard, PodState, PodState], float],
                 model=None,
                 discretizer: ActionDiscretizer = ActionDiscretizer(),
                 vectorizer: Vectorizer = V6()):
        super().__init__(board, vectorizer, discretizer)

        self.reward_func = reward_func
        self.depth = 0

        if model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    48,
                    input_shape=(self.vectorizer.vec_len(),),
                    activation="sigmoid",
                ),
                tf.keras.layers.Dense(
                    32,
                    activation="sigmoid",
                ),
                tf.keras.layers.Dense(
                    24,
                    activation="sigmoid",
                ),
                tf.keras.layers.Dense(
                    self.ad.num_actions,
                ),
            ])
        else:
            self.model = model

        self.model.compile(
            optimizer=tf.optimizers.Adam(0.001),
            metrics=['accuracy'],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

    def __wrap_reward_func(self) -> Callable[[PodBoard, PodState, PodState], float]:
        """
        Get a reward func that calculates the reward after playing to the current depth
        """
        def r_func(board: PodBoard, prev_pod, pod) -> float:
            for x in range(self.depth):
                prev_pod = pod
                pod = board.step(pod, self.play(pod))
            return self.reward_func(board, prev_pod, pod)

        return r_func

    def __build_labels(self, pods: List[Tuple[PodState, List[float]]], r_func, n_proc: int = 2) -> List[int]:
        """
        Using the given reward func, get labels (target actions) for each state
        """
        data = np.array_split(pods, n_proc)

        with Pool(n_proc) as p:
            results = p.starmap(_build_labels, [(self.ad, self.board, d, r_func) for d in data])
            labels = []
            for r in results: labels += r
            p.join()
            p.close()
            return labels


    def train(self, pods: List[Tuple[PodState, List[float]]], epochs: int):
        r_func = self.__wrap_reward_func()

        # For each pod, the state is simply the vectorized version of itself...
        states = np.array([t[1] for t in pods])

        print("Generating labels for {} pods...".format(len(pods)))
        start = perf_counter()
        # The label (target output) is whatever the best predicted tree branch is
        labels = np.array(self.__build_labels(pods, r_func))
        end = perf_counter()
        print("Labels generated in %.3f seconds" % (end - start))

        print("Training...")
        results = self.train_with_labels(states, labels, epochs)

        self.depth += 1
        return results

    def train_with_labels(self,
                          states,
                          labels,
                          epochs: int):
        return self.model.fit(states, labels, epochs=epochs, callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=5, min_delta=0.001)
        ])
