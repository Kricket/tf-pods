# Maximum speed that a pod can attain through normal acceleration (tested empirically)
import math
from typing import Tuple, List

from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import PlayOutput
from pod.util import PodState, clean_angle
from vec2 import Vec2, EPSILON, UNIT

MAX_VEL = 558
# Distance to use for scaling inputs
MAX_DIST = Vec2(Constants.world_x(), Constants.world_y()).length()

THRUST_VALUES = 3
ANGLE_VALUES = 6
MAX_ACTION = THRUST_VALUES * ANGLE_VALUES - 1

THRUST_INC = Constants.max_thrust() / (THRUST_VALUES - 1)
ANGLE_INC = Constants.max_turn() * 2 / (ANGLE_VALUES - 1)


def play_to_action(thrust: int, angle: float) -> int:
    """
    Given a legal play (angle/thrust), find the nearest discrete action
    """
    thrust_pct = thrust / Constants.max_thrust()
    angle_pct = (angle + Constants.max_turn()) / (2 * Constants.max_turn())
    thrust_idx = math.floor(thrust_pct * (THRUST_VALUES - 1))
    angle_idx = math.floor(angle_pct * (ANGLE_VALUES - 1))
    return math.floor(thrust_idx * ANGLE_VALUES + angle_idx)


def action_to_play(action: int) -> Tuple[int, float]:
    """
    Convert an action (in [0, THRUST_VALUES * ANGLE_VALUES - 1]) into the thrust, angle to play
    """
    # An integer in [0, THRUST_VALUES - 1]
    thrust_idx = int(action / ANGLE_VALUES)
    # An integer in [0, ANGLE_VALUES - 1]
    angle_idx = action % ANGLE_VALUES
    return thrust_idx * THRUST_INC, angle_idx * ANGLE_INC - Constants.max_turn()


def action_to_output(action: int, pod_angle: float, pod_pos: Vec2, po: PlayOutput = PlayOutput()) -> PlayOutput:
    """
    Convert an integer action to a PlayOutput for the given pod state
    """
    (thrust, rel_angle) = action_to_play(action)
    po.thrust = thrust

    real_angle = rel_angle + pod_angle
    real_dir = UNIT.rotate(real_angle) * 1000
    po.target = pod_pos + real_dir

    return po


def reward(pod: PodState, board: PodBoard) -> int:
    """
    Calculate the reward value for the given pod on the given board
    """
    # Reward for distance to next check - in [0, 1]
    dist_sq_to_check = (board.checkpoints[pod.nextCheckId] - pod.pos).square_length()
    world_approx = Constants.world_x() * Constants.world_y()
    dist_reward = max((world_approx - dist_sq_to_check) / world_approx, 0)

    # Bonus: 1 for each checkpoint already hit
    check_bonus = pod.nextCheckId + (pod.laps * len(board.checkpoints))

    return check_bonus + 2 * dist_reward


def state_to_vector_old(
        pod_pos: Vec2,
        pod_vel: Vec2,
        pod_angle: float,
        target_check: Vec2,
        next_check: Vec2
) -> List[float]:
    """
    Transform the given pod state information into a simple array that can be fed as input to a NN
    """
    # All values here are in the game frame of reference. We do the rotation at the end.
    vel_length = pod_vel.length()
    vel_angle = math.acos(pod_vel.x / vel_length) if vel_length > EPSILON else 0.0

    pod_to_check1 = target_check - pod_pos
    dist_to_check1 = pod_to_check1.length()
    ang_to_check1 = math.acos(pod_to_check1.x / dist_to_check1)

    check1_to_check2 = next_check - target_check
    dist_check1_to_check2 = check1_to_check2.length()
    ang_check1_to_check2 = math.acos(
        (check1_to_check2 * pod_to_check1) / (dist_check1_to_check2 * dist_to_check1)
    )

    # Re-orient so pod is at (0,0) angle 0.0
    return [
        # Angle between pod and its velocity
        clean_angle(vel_angle - pod_angle),
        # Angle between pod orientation and target check
        clean_angle(ang_to_check1 - pod_angle),
        # Angle between (pod to target check) and (target check to next check)
        clean_angle(ang_check1_to_check2 - ang_to_check1 - pod_angle),
        # Scaled velocity
        vel_length / MAX_VEL,
        # Scaled distance to target check
        dist_to_check1 / MAX_DIST,
        # Scaled distance between next 2 checks
        dist_check1_to_check2 / MAX_DIST
    ]

def state_to_vector(
        pod_pos: Vec2,
        pod_vel: Vec2,
        pod_angle: float,
        target_check: Vec2,
        next_check: Vec2
) -> List[float]:
    # Velocity is already relative to the pod, so it just needs to be rotated
    vel = pod_vel.rotate(-pod_angle)
    check1 = (target_check - pod_pos).rotate(-pod_angle)
    check2 = (next_check - pod_pos).rotate(-pod_angle)

    return [vel.x, vel.y, check1.x, check1.y, check2.x, check2.y]
