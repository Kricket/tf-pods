import math
from typing import Tuple, List

import numpy as np
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import PlayOutput
from pod.game import game_step
from pod.util import PodState, clean_angle
from vec2 import Vec2, UNIT

# Maximum speed that a pod can attain through normal acceleration (tested empirically)
MAX_VEL = 558
# Distance to use for scaling inputs
MAX_DIST = Vec2(Constants.world_x(), Constants.world_y()).length()


class ActionDiscretizer:
    def __init__(self, num_thrust: int = 3, num_angle: int = 3):
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

    def get_best_action(self, board: PodBoard, pod: PodState) -> int:
        """
        Get the action that will result in the highest reward for the given state
        """
        best_action = 0
        best_reward = -999

        for action in range(self.num_actions):
            next_state = PodState()
            game_step(board, pod, self.action_to_output(action, pod.angle, pod.pos), next_state)
            r = reward(next_state, board)
            if r > best_reward:
                best_reward = r
                best_action = action

        return best_action


def reward(pod: PodState, board: PodBoard) -> int:
    """
    Calculate the reward value for the given pod on the given board
    """
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos

    # Reward for distance to next check - in [0, 1]
    dist_to_check = pod_to_check.length()
    dist_penalty = dist_to_check / MAX_DIST

    # Bonus: points for each checkpoint already hit
    check_bonus = pod.nextCheckId + (pod.laps * len(board.checkpoints))

    # Bonus: a tiny amount if the pod is pointing at the next check (helps to distinguish between
    # states with 0 thrust)
    ang_diff = clean_angle(pod_to_check.angle() - pod.angle)
    ang_bonus = 0.1 * (math.pi - math.fabs(ang_diff)) / math.pi

    return ang_bonus + check_bonus - dist_penalty + 1


def gen_pods(
        checkpoint: Vec2,
        xy_list: List[int],
        ang_list: List[float],
        vel_ang_list: List[float],
        vel_mag_list: List[float]
):
    pods = []

    for x in xy_list:
        for y in xy_list:
            abs_pos = Vec2(x, y)
            ang_to_check = abs_pos.angle() + math.pi
            pos = abs_pos + checkpoint
            for a in ang_list:
                angle = clean_angle(ang_to_check + a)
                for va in vel_ang_list:
                    vel_dir = UNIT.rotate(ang_to_check + va)
                    for vm in vel_mag_list:
                        vel = vel_dir * vm

                        pod = PodState(pos)
                        pod.angle = angle
                        pod.vel = vel
                        pods.append(pod)

    np.random.shuffle(pods)
    print("{} pods generated".format(len(pods)))
    return pods
