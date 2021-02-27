from typing import Callable

from pod.ai.action_discretizer import ActionDiscretizer
from pod.board import PodBoard
from pod.controller import Controller, PlayOutput
from pod.util import PodState


class RewardController(Controller):
    """
    A Controller that always takes the action that will produce the highest reward
    """
    def __init__(self,
                 board: PodBoard,
                 reward_func: Callable[[PodBoard, PodState, PodState], float],
                 discretizer: ActionDiscretizer = ActionDiscretizer(3, 3)):
        super().__init__(board)
        self.reward_func = reward_func
        self.ad = discretizer

    def play(self, pod: PodState) -> PlayOutput:
        rewards = []
        best_reward = -999
        best_play = None
        for action in range(0, self.ad.num_actions):
            play = self.ad.action_to_output(action, pod.angle, pod.pos)
            next_pod = self.board.step(pod, play)
            reward = self.reward_func(self.board, pod, next_pod)
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_play = play

        return best_play
