import random
from time import perf_counter
from typing import List, Tuple

from pod.ai.action_discretizer import DiscreteActionController, ActionDiscretizer
from pod.ai.greedy_controller import GreedyController
from pod.ai.rewards import RewardFunc, re_dcat, re_dca, re_cts, speed_reward
from pod.board import PodBoard
from pod.util import PodState


class MCTSController(DiscreteActionController):
    def __init__(self,
                 board: PodBoard,
                 depth: int = 10,
                 calc_time: float = 1.0,
                 reward_func: RewardFunc = re_dcat,
                 ad: ActionDiscretizer = ActionDiscretizer()):
        super().__init__(board, ad)
        self.reward_func = reward_func
        self.root = None
        self.last_action = None
        self.depth = depth
        self.calc_time = calc_time
        _Node.con = self

    def reset(self):
        self.root = None

    def get_action(self, pod: PodState) -> int:
        if self.root is None or pod != self.root.pod:
            self.root = _Node(pod)

        cur_score = self.reward_func(self.board, pod)

        # As long as there's time left, explore to the given depth
        count = 0
        end = start = perf_counter()
        while (end - start) < self.calc_time:
            self.root.expand(self.depth, cur_score)
            count += 1
            end = perf_counter()

        action = self.root.best_action
        self.root = self.root.children[action]

        return action


###############################################################################


class _Node:
    con: MCTSController = None

    def __init__(self, pod: PodState):
        self.pod = pod
        self.children = [None for x in range(_Node.con.ad.num_actions)]
        self.best_action = -1
        self.best_score = -9999999

    def expand(self, depth: int, min_score: float) -> float:
        """
        Perform one MC run to the given depth:
        - randomly pick children until arriving at the depth
        - Greedy playout (up to the next check)
        :param min_score:
        :param depth: How deep to explore
        :return: The score of the best leaf node found
        """
        if depth > 1:
            action = self.__pick_child()
            score = self.children[action].expand(depth - 1, min_score)
            if score > self.best_score:
                self.best_score = score
                self.best_action = action
            return self.best_score
        else:
            return self.__playout(min_score)

    def __pick_child(self) -> int:
        action = random.randint(1, _Node.con.ad.num_actions) - 1
        if self.children[action] is None:
            self.children[action] = _Node(_Node.con.board.step(
                self.pod,
                _Node.con.ad.action_to_output(action, self.pod.angle, self.pod.pos))
            )

        return action

    def __playout(self, min_score) -> float:
        """
        Play forward until reaching a "terminal" state. At that point, return the score.
        """
        score = _Node.con.reward_func(_Node.con.board, self.pod)
        # Don't bother if we're doing badly
        if score < min_score: return score

        con = GreedyController(_Node.con.board, speed_reward)
        pod = _Node.con.board.step(self.pod, con.play(self.pod))
        while pod.nextCheckId == self.pod.nextCheckId:
            pod = _Node.con.board.step(pod, con.play(pod))

        return _Node.con.reward_func(_Node.con.board, pod)

