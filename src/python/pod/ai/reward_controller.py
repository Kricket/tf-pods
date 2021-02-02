from pod.ai.ai_utils import calc_reward, ActionDiscretizer
from pod.board import PodBoard
from pod.controller import Controller, PlayInput, PlayOutput
from pod.game import game_step


class RewardController(Controller):
    """
    A Controller that always takes the action that will produce the highest reward
    """
    def __init__(self, board: PodBoard, discretizer: ActionDiscretizer = ActionDiscretizer(3, 3)):
        self.board = board
        self.ad = discretizer

    def play(self, pi: PlayInput) -> PlayOutput:
        rewards = []
        best_reward = -999
        best_play = None
        for action in range(0, self.ad.num_actions):
            play = self.ad.action_to_output(action, pi.angle, pi.pos)
            pod = pi.as_pod()
            game_step(self.board, pod, play, pod)
            reward = calc_reward(pod, self.board)
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_play = play

        return best_play
