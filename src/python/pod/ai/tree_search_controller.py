from typing import Callable, Tuple

from pod.ai.action_discretizer import ActionDiscretizer
from pod.board import PodBoard, PlayOutput
from pod.controller import Controller
from pod.util import PodState


class _Node:
    def __init__(self, pod: PodState):
        self.pod = pod
        self.children = None
        self.score = None

    def expand(self,
               ad: ActionDiscretizer,
               board: PodBoard,
               reward_func: Callable[[PodBoard, PodState, PodState], float],
               max_depth: int):
        if self.children is None:
            # Generate immediate children
            self.children = [
                _Node(board.step(
                    self.pod,
                    ad.action_to_output(action, self.pod.angle, self.pod.pos)
                ))
                for action in range(ad.num_actions)
            ]

        if max_depth > 1:
            # Generate the next layer of children...
            for child in self.children:
                child.expand(ad, board, reward_func, max_depth - 1)
        else:
            # ...or if we're done, calculate the children's scores
            for child in self.children:
                child.score = reward_func(board, self.pod, child.pod)

        # Now, set our own score to the best child score
        self.score = max(child.score for child in self.children)

    def best_child(self) -> Tuple[int, '_Node']:
        """
        Get the best (highest score) child node, with its index
        """
        idx = 0
        child = self.children[0]
        for i in range(1, len(self.children)):
            if self.children[i].score > child.score:
                idx = i
                child = self.children[i]

        return idx, child


class TreeSearchController(Controller):
    def __init__(self,
                 board: PodBoard,
                 reward_func: Callable[[PodBoard, PodState, PodState], float],
                 max_depth: int = 4,
                 ad: ActionDiscretizer = ActionDiscretizer(3, 3)):
        super().__init__(board)
        self.ad = ad
        self.max_depth = max_depth
        self.reward_func = reward_func
        self.root = None

    def play(self, pod: PodState) -> PlayOutput:
        # Explore ahead
        if self.root is None or pod != self.root.pod:
            print("Root is not equal to current state - rebuilding...")
            self.root = _Node(pod)
        self.root.expand(self.ad, self.board, self.reward_func, self.max_depth)

        action, node = self.root.best_child()

        self.root = node

        return self.ad.action_to_output(action, pod.angle, pod.pos)
