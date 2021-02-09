import math

from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState, clean_angle


def dense_reward(pod: PodState, board: PodBoard) -> float:
    """
    Dense reward: gives partial credit based on distance to next check
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
    ang_bonus = 0.01 * (math.pi - math.fabs(ang_diff)) / math.pi

    return ang_bonus + 3*check_bonus - dist_penalty + 1

def diff_reward(prev_pod: PodState, next_pod: PodState, board: PodBoard) -> float:
    """
    Also gives dense rewards, but based on the difference in check distance
    instead of absolute distance
    """
    if prev_pod.nextCheckId > next_pod.nextCheckId:
        # Distance would make no sense here, so instead just give a big bonus
        return 5

    check = board.checkpoints[prev_pod.nextCheckId]
    prev_dist = (check - prev_pod.pos).length()
    next_dist = (check - next_pod.pos).length()
    return (next_dist - prev_dist) / Constants.max_vel()
