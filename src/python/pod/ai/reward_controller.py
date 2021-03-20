from pod.ai.action_discretizer import ActionDiscretizer, DiscreteActionController
from pod.ai.rewards import RewardFunc
from pod.board import PodBoard
from pod.util import PodState


class RewardController(DiscreteActionController):
    """
    A Controller that always takes the action that will produce the highest reward
    """
    def __init__(self,
                 board: PodBoard,
                 reward_func: RewardFunc,
                 discretizer: ActionDiscretizer = ActionDiscretizer()):
        super().__init__(board, discretizer)
        self.reward_func = reward_func

    def get_action(self, pod: PodState) -> int:
        return self.ad.get_best_action(self.board, pod, self.reward_func)