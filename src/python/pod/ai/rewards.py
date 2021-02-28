import math
from typing import Tuple, Callable, List

from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState, clean_angle

#################################################
# Reward functions: signature is
# func(board, prev_state, next_state) -> float
#################################################


def make_reward(
        factors: List[Tuple[float, Callable[[PodBoard, PodState, PodState], float]]]
) -> Callable[[PodBoard, PodState, PodState], float]:
    """
    Generate a reward function which is a linear combination of other reward functions
    """
    def rfunc(board: PodBoard, prev_pod: PodState, pod: PodState):
        total = 0.0
        for (factor, func) in factors:
            res = factor * func(board, prev_pod, pod)
            total += res
        return total
    return rfunc


def dist_reward(board: PodBoard, prev_pod: PodState, pod: PodState) -> float:
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


def ang_reward(board: PodBoard, prev_pod: PodState, pod: PodState) -> float:
    """
    Returns the angle between the pod's direction and the next checkpoint, scaled to be in (0, 1)
    """
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos
    angle = clean_angle(pod_to_check.angle() - pod.angle)

    return 1 - math.fabs(angle / math.pi)


def check_reward(board: PodBoard, prev_pod: PodState, pod: PodState) -> float:
    """
    Returns 1 point for every checkpoint hit
    """
    return len(board.checkpoints) * pod.laps + pod.nextCheckId


def diff_reward(board: PodBoard, prev_pod: PodState, next_pod: PodState) -> float:
    """
    Dense reward based on the change in distance to the next check, scaled to be in (-1, 1).

    This should be used with the check_reward, since it doesn't know what to do when
      the pod hits a check.
    """
    if prev_pod.nextCheckId != next_pod.nextCheckId:
        # Distance would make no sense here.
        return 0

    check = board.checkpoints[prev_pod.nextCheckId]
    prev_dist = (check - prev_pod.pos).length()
    next_dist = (check - next_pod.pos).length()
    return (prev_dist - next_dist) / Constants.max_vel()


def speed_reward(board: PodBoard, prev_pod: PodState, next_pod: PodState) -> float:
    """
    Indicates how much the speed is taking us toward the next check (scaled).

    Similar to the diff reward, although the score
    """
    pod_to_check = board.checkpoints[next_pod.nextCheckId] - next_pod.pos
    dist_to_check = pod_to_check.length()

    # a*b = |a|*|b|*cos
    # Thus, vel*check / dist = how much the vel is taking us toward the check
    return (next_pod.vel * pod_to_check) / (dist_to_check * Constants.max_vel())

