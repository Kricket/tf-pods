import math
from typing import Tuple

from pod.ai.rewards import RewardFunc
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import PlayOutput, Controller
from pod.util import PodState
from vec2 import Vec2, UNIT


class ActionDiscretizer:
    """
    Tools to manage chopping up the action space into discrete segments
    """
    def __init__(self, num_thrust: int = 2, num_angle: int = 3):
        self.num_thrust = num_thrust
        self.num_angle = num_angle
        self.num_actions = num_thrust * num_angle

        self._thrust_inc = Constants.max_thrust() / (self.num_thrust - 1)
        self._angle_inc = Constants.max_turn() * 2 / (self.num_angle - 1)

    def play_to_action(self, thrust: int, angle: float) -> int:
        """
        Given a legal play (angle/thrust), find the nearest discrete action
        """
        thrust_pct = thrust / Constants.max_thrust()
        angle_pct = (angle + Constants.max_turn()) / (2 * Constants.max_turn())
        thrust_idx = math.floor(thrust_pct * (self.num_thrust - 1))
        angle_idx = math.floor(angle_pct * (self.num_angle - 1))
        return math.floor(thrust_idx * self.num_angle + angle_idx)

    def action_to_play(self, action: int) -> Tuple[int, float]:
        """
        Convert an action (in [0, THRUST_VALUES * ANGLE_VALUES - 1]) into the thrust, angle to play
        """
        # An integer in [0, THRUST_VALUES - 1]
        thrust_idx = int(action / self.num_angle)
        # An integer in [0, ANGLE_VALUES - 1]
        angle_idx = action % self.num_angle
        return thrust_idx * self._thrust_inc, angle_idx * self._angle_inc - Constants.max_turn()

    def action_to_output(self, action: int, pod_angle: float, pod_pos: Vec2, po: PlayOutput = None) -> PlayOutput:
        """
        Convert an integer action to a PlayOutput for the given pod state
        """
        if po is None:
            po = PlayOutput()

        (thrust, rel_angle) = self.action_to_play(action)
        po.thrust = thrust

        real_angle = rel_angle + pod_angle
        real_dir = UNIT.rotate(real_angle) * 1000
        po.target = pod_pos + real_dir

        return po

    def get_best_action(self,
                        board: PodBoard,
                        pod: PodState,
                        reward_func: RewardFunc
                        ) -> int:
        """
        Get the action that will result in the highest reward for the given state
        """
        best_action = 0
        best_reward = -999

        for action in range(self.num_actions):
            play = self.action_to_output(action, pod.angle, pod.pos)
            next_pod = board.step(pod, play)
            reward = reward_func(board, next_pod)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action

class DiscreteActionController(Controller):
    """
    A Controller that chooses a play from a discretized subset of the Action space
    """
    def __init__(self, board: PodBoard, ad: ActionDiscretizer):
        super().__init__(board)
        self.ad = ad

    def get_action(self, pod: PodState) -> int:
        """
        Get the discrete action value for the play that this Controller will make
        """
        pass

    def play(self, pod: PodState) -> PlayOutput:
        return self.ad.action_to_output(self.get_action(pod), pod.angle, pod.pos)
