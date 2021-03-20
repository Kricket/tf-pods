from typing import Tuple, Dict

from pod.ai.action_discretizer import ActionDiscretizer, DiscreteActionController
from pod.ai.rewards import RewardFunc
from pod.board import PodBoard
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

    def __make_child(self, action: int):
        child = _Node(self.ts.board.step(
            self.pod,
            self.ts.ad.action_to_output(action, self.pod.angle, self.pod.pos)
        ), self.ts)

        return child

    def expand(self, to_depth: int):
        if self.children is None:
            # Generate immediate children
            self.children = [self.__make_child(action)
                             for action in range(self.ts.ad.num_actions)]

        if to_depth > 1:
            # Generate the next layer of children...
            for child in self.children:
                child.expand(to_depth - 1)
        else:
            # Our children are leaf nodes. Evaluate them.
            for child in self.children:
                child.score = self.ts.reward_func(self.ts.board, child.pod)

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

#################################################

class TreeSearchController(DiscreteActionController):
    def __init__(self,
                 board: PodBoard,
                 reward_func: RewardFunc,
                 max_depth: int = 4,
                 ad: ActionDiscretizer = ActionDiscretizer()):
        super().__init__(board, ad)
        self.max_depth = max_depth
        self.reward_func = reward_func
        self.root = None
        self.last_action = None

    def reset(self):
        self.root = None

    def get_action(self, pod: PodState) -> int:
        # Explore ahead
        if self.root is None or pod != self.root.pod:
            self.root = _Node(pod, self)
        self.root.expand(self.max_depth)

        action, node = self.root.best_child()

        self.root = node
        self.last_action = action

        return action

    def record(self, log: Dict):
        super().record(log)
        log['action'] = self.last_action
