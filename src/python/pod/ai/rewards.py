import math
from typing import Tuple, Callable, List

from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState, clean_angle

#################################################
# Reward functions: signature is
# func(board, state) -> float
#################################################

RewardFunc = Callable[[PodBoard, PodState], float]

def pgr(board: PodBoard, pod: PodState) -> float:
    """
    Pretty Good Reward
    Attempts to estimate the distance without using a SQRT calculation.
    """
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos
    prev_to_next_check = board.checkpoints[pod.nextCheckId] - board.get_check(pod.nextCheckId - 1)
    pod_dist_estimate = (math.fabs(pod_to_check.x) + math.fabs(pod_to_check.y)) / 2
    check_dist_estimate = (math.fabs(prev_to_next_check.x) + math.fabs(prev_to_next_check.y)) / 2
    dist_estimate = pod_dist_estimate / check_dist_estimate

    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId

    return 2*checks_hit - dist_estimate + 1

def regood(board: PodBoard, pod: PodState) -> float:
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos
    prev_to_check = board.checkpoints[pod.nextCheckId] - board.get_check(pod.nextCheckId-1)

    # This scales, not by a fixed MAX_DIST, but relative to the distance between the checks.
    # So right after hitting a check, this should be about 1 (slightly off since we hit the
    # edge of the check, not the center)
    dist_penalty = math.sqrt(pod_to_check.square_length() / prev_to_check.square_length())

    # Bonus for each check hit. By making it 2 per check, we ensure that the reward is always
    # higher after hitting a check. (If left at 1, the dist_penalty could be slightly greater
    # than 1, leading to a DECREASE in reward for hitting a check)
    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId

    return 2 * checks_hit + 1 - dist_penalty


#################################################
# These are for experimenting...
#################################################

def make_reward(
        factors: List[Tuple[float, RewardFunc]]
) -> RewardFunc:
    """
    Generate a reward function which is a linear combination of other reward functions
    """
    def rfunc(board: PodBoard, pod: PodState):
        total = 0.0
        for (factor, func) in factors:
            res = factor * func(board, pod)
            total += res
        return total
    return rfunc


def dist_reward(board: PodBoard, pod: PodState) -> float:
    """
    Dense reward based on the distance to the next checkpoint, scaled to be in (0, 1).

    In general, the difference in output between two consecutive turns is very slight.
    In addition, hitting a check produces a big PENALTY, since the distance suddenly increases!
    """
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos

    # Reward for distance to next check - in [0, 1]
    dist_to_check = pod_to_check.length()
    dist_penalty = dist_to_check / MAX_DIST

    return 1 - dist_penalty


def ang_reward(board: PodBoard, pod: PodState) -> float:
    """
    Returns the angle between the pod's direction and the next checkpoint, scaled to be in (0, 1)
    """
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos
    angle = clean_angle(pod_to_check.angle() - pod.angle)

    return 1 - math.fabs(angle / math.pi)


def check_reward(board: PodBoard, pod: PodState) -> float:
    """
    Sparse reward: returns 1 point for every checkpoint hit
    """
    return len(board.checkpoints) * pod.laps + pod.nextCheckId


def speed_reward(board: PodBoard, next_pod: PodState) -> float:
    """
    Indicates how much the speed is taking us toward the next check (scaled).
    """
    pod_to_check = board.checkpoints[next_pod.nextCheckId] - next_pod.pos
    dist_to_check = pod_to_check.length()

    # a*b = |a|*|b|*cos
    # Thus, vel*check / dist = how much the vel is taking us toward the check
    return (next_pod.vel * pod_to_check) / (dist_to_check * Constants.max_vel())
