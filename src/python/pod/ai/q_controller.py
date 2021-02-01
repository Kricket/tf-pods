import math
import random
from typing import List, Union, Generator

import numpy as np

from pod.ai.ai_utils import MAX_VEL, reward
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput, PlayInput

###############################################
# Discretized state space
from pod.game import game_step
from pod.util import PodState
from vec2 import UNIT, Vec2

# Square of distance from pod to next check
DIST_SQ_STATES = [a**2 for a in [
    *[x * Constants.check_radius() for x in range(1, 7)],
    *[x * Constants.check_radius() for x in range(8, 16, 2)]
]]

# Angle from pod to next check
ANG_STATES = [
    math.pi * (-0.5),
    math.pi * (-0.3),
    math.pi * (-0.2),
    math.pi * (-0.1),
    math.pi * (-0.05),
    0,
    math.pi * 0.05,
    math.pi * 0.1,
    math.pi * 0.2,
    math.pi * 0.3,
    math.pi * 0.5,
]

# Angle from pod to its velocity
VEL_ANG_STATES = [s for s in ANG_STATES]

# Magnitude squared of pod velocity
VEL_MAG_SQ_STATES = [v ** 2 for v in [x * MAX_VEL / 5 for x in range(1, 5)]]

TOTAL_STATES = len(DIST_SQ_STATES) * len(ANG_STATES) * len(VEL_ANG_STATES) * len(VEL_MAG_SQ_STATES)


def get_index(value: float, table: List[float]) -> int:
    """
    Get the index of the value in the table that is closest to the given value.
    Assumes that the table is sorted in increasing order!
    """
    if value < table[0]: return 0
    if value > table[-1]: return len(table) - 1

    val_dist = math.fabs(value - table[0])
    for idx in range(1, len(table)):
        next_dist = math.fabs(value - table[idx])
        if next_dist < val_dist:
            val_dist = next_dist
        else:
            return idx - 1
    return len(table) - 1


def pod_to_state(pod: Union[PlayInput, PodState], board: PodBoard) -> int:
    """
    Get the state ID for the given pod
    """
    pod_to_check = board.get_check(pod.nextCheckId) - pod.pos
    dist_state = get_index(pod_to_check.square_length(), DIST_SQ_STATES)
    ang_state = get_index(pod_to_check.angle() - pod.angle, ANG_STATES)
    vel_ang = pod.vel.angle() - pod.angle
    vel_ang_state = get_index(vel_ang, VEL_ANG_STATES)
    vel_mag_state = get_index(pod.vel.square_length(), VEL_MAG_SQ_STATES)

    return dist_state    * (len(VEL_MAG_SQ_STATES) * len(VEL_ANG_STATES) * len(ANG_STATES)) \
         + ang_state     * (len(VEL_MAG_SQ_STATES) * len(VEL_ANG_STATES)) \
         + vel_ang_state *  len(VEL_MAG_SQ_STATES) \
         + vel_mag_state


###############################################
# Discretized action space
# Here, the actions are to aim for various points around the target

TARGET_ACTIONS = [UNIT.rotate(math.pi * a / 6.0) * Constants.check_radius() * 5 for a in range(0, 12)]
THRUST_ACTIONS = [0, Constants.max_thrust() / 2, Constants.max_thrust()]

TOTAL_ACTIONS = len(TARGET_ACTIONS) * len(THRUST_ACTIONS)

def action_to_play(action: int, next_check: Vec2) -> PlayOutput:
    thrust_idx = int(action / len(TARGET_ACTIONS))
    target_idx = action % len(TARGET_ACTIONS)
    return PlayOutput(next_check + TARGET_ACTIONS[target_idx], THRUST_ACTIONS[thrust_idx])


class QController(Controller):
    """
    A Controller that uses Q-Learning to win the race. The state and action spaces are discretized
    so that the table is manageable.
    """
    def __init__(self, board: PodBoard):
        self.board = board
        self.q_table = np.zeros((TOTAL_STATES, TOTAL_ACTIONS))

    def play(self, pi: PlayInput) -> PlayOutput:
        state = pod_to_state(pi, self.board)
        action = np.argmax(self.q_table[state,:])
        return action_to_play(action, pi.nextCheck)

    def train(self,
              num_episodes: int = 10,
              max_turns: int = 50,
              prob_rand_action: float = 0.5,
              learning_rate: float = 0.5,
              future_discount: float = 0.9):
        reward_per_ep = []

        for episode in range(num_episodes):
            total_ep_reward = 0
            # The pod starts in a random position at a fixed (far) distance from the check,
            # pointing in a random direction
            pos_offset = UNIT.rotate(random.random() * 2 * math.pi) * Constants.check_radius() * 17
            pod = PodState(
                pos=self.board.get_check(0) + pos_offset,
                angle=2 * math.pi * random.random() - math.pi
            )
            current_state = pod_to_state(pod, self.board)

            for turn in range(max_turns):
                # Choose an action
                if random.random() < prob_rand_action:
                    action = math.floor(random.random() * TOTAL_ACTIONS)
                else:
                    action = np.argmax(self.q_table[current_state,:])

                # Take the action and calculate the reward. Since the discretization of the state space is
                # rough, we repeat the action until we get to a new state
                next_state = current_state
                play = action_to_play(action, self.board.get_check(pod.nextCheckId))
                tries = 0
                while next_state == current_state and tries < 5:
                    game_step(self.board, pod, play, pod)
                    next_state = pod_to_state(pod, self.board)
                    tries += 1

                r = reward(pod, self.board)

                # Update the Q-table
                self.q_table[current_state, action] = (1 - learning_rate) * self.q_table[current_state, action] \
                    + learning_rate * (r + future_discount * max(self.q_table[next_state,:]))

                total_ep_reward += r
                current_state = next_state

            reward_per_ep.append(total_ep_reward)

        return reward_per_ep
