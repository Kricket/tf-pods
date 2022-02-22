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
DIST_BASE = math.sqrt(Constants.world_x() * Constants.world_y())

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

###############################################################################
# Pre-baked rewards
#
# d = distance - is important because it rewards getting closer to the check
# c = checks - is important because distance suddenly gets worse after hitting a check
# a = angle - a tiny amount to nudge the agent to turn toward the check
# t = turns - penalty for sloth: hurry up!
#
###############################################################################

def re_dc(board: PodBoard, pod: PodState) -> float:
    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId

    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos
    dist_penalty = pod_to_check.length() / DIST_BASE

    return 3 * (checks_hit + 1) - dist_penalty

def re_dca(board: PodBoard, pod: PodState) -> float:
    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId

    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos

    angle = math.fabs(clean_angle(pod_to_check.angle() - pod.angle))
    a_penalty = (angle / math.pi) / 10 if angle > Constants.max_turn() else 0

    dist_penalty = pod_to_check.length() / DIST_BASE

    return 3 * (checks_hit + 1) - dist_penalty - a_penalty

def re_dcat(board: PodBoard, pod: PodState) -> float:
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos

    # Scaled distance to next check
    dist_penalty = pod_to_check.length() / DIST_BASE

    # Bonus for each check hit. By making it 2 per check, we ensure that the reward is always
    # higher after hitting a check. (If left at 1, the dist_penalty could be slightly greater
    # than 1, leading to a DECREASE in reward for hitting a check)
    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId

    # A tiny bit for the angle. This should really be tiny - its purpose is to serve as a
    # tie-breaker (to prevent the pod from going into orbit around a check).
    angle = math.fabs(clean_angle(pod_to_check.angle() - pod.angle))
    a_penalty = (angle / math.pi) / 10 if angle > Constants.max_turn() else 0

    # And finally: this can be important to prevent agents from doing nothing.
    # The reduction factor is slightly more than the number of turns it takes
    # (on average) to get from one check to another
    turn_penalty = pod.turns / 20

    return 3 * (checks_hit + 1) \
           - dist_penalty \
           - a_penalty \
           - turn_penalty

def check_and_speed_reward(board: PodBoard, pod: PodState) -> float:
    """
    Like check_reward, but if past the first checkpoint, also adds the speed_reward.

    This is mostly useful for testing/training, in situations where the simulation
    stops upon hitting the first check.
    """
    rew = check_reward(board, pod)
    if rew > 0:
        rew += speed_reward(board, pod)
    return rew

def re_cts(board: PodBoard, pod: PodState) -> float:
    checks_hit = len(board.checkpoints) * pod.laps + pod.nextCheckId
    turn_penalty = pod.turns / 20
    speed = speed_reward(board, pod)

    return checks_hit + speed - turn_penalty


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
    # Thus, vel*check / dist = vel component going towards the check
    return (next_pod.vel * pod_to_check) / (dist_to_check * Constants.max_vel())
