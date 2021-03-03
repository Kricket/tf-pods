import numpy as np

from typing import Callable, Tuple, List

from pod.ai.action_discretizer import ActionDiscretizer
from pod.board import PodBoard, PlayOutput
from pod.controller import Controller
from pod.util import PodState


#################################################

class _Node:
    """
    A node in the Tree of moves being searched
    """
    def __init__(self, pod: PodState, ts: 'TreeSearchController'):
        self.pod = pod
        self.ts = ts
        self.children = None
        self.score = None
        self.reward = None

    def __make_child(self, action: int):
        child = _Node(self.ts.board.step(
            self.pod,
            self.ts.ad.action_to_output(action, self.pod.angle, self.pod.pos)
        ), self.ts)

        child.reward = child.score = self.ts.reward_func(
            self.ts.board,
            self.pod,
            child.pod
        )

        return child

    def expand(self, to_depth: int):
        if self.children is None:
            # Generate immediate children
            self.children = [self.__make_child(action)
                             for action in range(self.ts.ad.num_actions)]
#            pruner(self.children)

        if to_depth > 1:
            # Generate the next layer of children...
            for child in self.children:
                if child is not None:
                    child.expand(to_depth - 1)

        # Now, set our own score to the best child score
        self.score = max(
            -99999 if child is None else child.score
            for child in self.children)

    def best_child(self) -> Tuple[int, '_Node', int]:
        """
        Get the best (highest score) child node, with its index
        """
        reward_order = np.argsort([
            -9999 if child is None else child.reward
            for child in self.children
        ])[::-1]

        idx = 0
        child = self.children[0]
        for i in range(1, len(self.children)):
            if self.children[i] is not None:
                if child is None or self.children[i].score > child.score:
                    idx = i
                    child = self.children[i]

        return idx, child, np.where(reward_order == idx)[0]

#################################################

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
        self.log = []
        self.check_turns = []

    def reset(self):
        self.log = []
        self.root = None

    def play(self, pod: PodState) -> PlayOutput:
        #if pod.turns % 10 == 0: print("Playing turn " + str(pod.turns))

        # Explore ahead
        if self.root is None or pod != self.root.pod:
            self.root = _Node(pod, self)
        self.root.expand(self.max_depth)

        action, node, act_order = self.root.best_child()
        self.log.append(act_order)
        if self.root.pod.nextCheckId != node.pod.nextCheckId:
            self.check_turns.append(node.pod.turns)

        self.root = node

        return self.ad.action_to_output(action, pod.angle, pod.pos)