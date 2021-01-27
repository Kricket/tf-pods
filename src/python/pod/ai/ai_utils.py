import math
from typing import Tuple, List, Generator

import numpy as np
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import PlayOutput
from pod.game import game_step
from pod.util import PodState, clean_angle
from vec2 import Vec2, UNIT

# Length of the state vector (model input)
STATE_VECTOR_LEN = 6

# Maximum speed that a pod can attain through normal acceleration (tested empirically)
MAX_VEL = 558
# Distance to use for scaling inputs
MAX_DIST = Vec2(Constants.world_x(), Constants.world_y()).length()

THRUST_VALUES = 3
ANGLE_VALUES = 3
NUM_ACTIONS = THRUST_VALUES * ANGLE_VALUES

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


def action_to_output(action: int, pod_angle: float, pod_pos: Vec2, po: PlayOutput = None) -> PlayOutput:
    """
    Convert an integer action to a PlayOutput for the given pod state
    """
    if po is None:
        po = PlayOutput()

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
    pod_to_check = board.checkpoints[pod.nextCheckId] - pod.pos

    # Reward for distance to next check - in [0, 1]
    dist_to_check = pod_to_check.length()
    dist_penalty = dist_to_check / MAX_DIST

    # Bonus: points for each checkpoint already hit
    checks_hit = pod.nextCheckId + (pod.laps * len(board.checkpoints))
    check_bonus = 5  * checks_hit

    # Bonus: a tiny amount if the pod is pointing at the next check (helps to distinguish between
    # states with 0 thrust)
    ang_diff = clean_angle(pod_to_check.angle() - pod.angle)
    ang_bonus = (math.pi - math.fabs(ang_diff)) / math.pi

    return ang_bonus + check_bonus - dist_penalty + 1


def state_to_vector(
        pod_pos: Vec2,
        pod_vel: Vec2,
        pod_angle: float,
        target_check: Vec2,
        next_check: Vec2
) -> List[float]:
    # Velocity is already relative to the pod, so it just needs to be rotated
    vel = pod_vel.rotate(-pod_angle) / MAX_VEL
    check1 = (target_check - pod_pos).rotate(-pod_angle) / MAX_DIST
    check2 = (next_check - pod_pos).rotate(-pod_angle) / MAX_DIST

    res = [vel.x, vel.y, check1.x, check1.y, check2.x, check2.y]
    return res


def frange(mini: float, maxi: float, steps: int):
    """
    Generate 'steps' values from mini to maxi (including both endpoints)
    """
    if steps <= 1:
        yield mini
    else:
        total = maxi - mini
        step_size = total / (steps - 1) # So that we include min and max
        for i in range(steps):
            yield mini + step_size * i


def gen_pods(
        checkpoint: Vec2,
        xy_range: Generator[float, None, None],
        ang_range: Generator[float, None, None],
        vel_ang_range: Generator[float, None, None],
        vel_mag_range: Generator[float, None, None]
):
    pods = []
    xy_list = list(xy_range)
    ang_list = list(ang_range)
    vel_ang_list = list(vel_ang_range)
    vel_mag_list = list(vel_mag_range)

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


def get_best_action(board: PodBoard, pod: PodState) -> int:
    """
    Get the action that will result in the highest reward for the given state
    """
    best_action = 0
    best_reward = -999

    for action in range(NUM_ACTIONS):
        next_state = PodState()
        game_step(board, pod, action_to_output(action, pod.angle, pod.pos), next_state)
        r = reward(next_state, board)
        if r > best_reward:
            best_reward = r
            best_action = action

    return best_action
