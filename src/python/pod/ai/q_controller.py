import math
import random
from typing import List, Callable, Tuple

import numpy as np
from pod.ai.action_discretizer import ActionDiscretizer
from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller, PlayOutput
from pod.util import PodState
from vec2 import UNIT


def _discretize(val: float, precision: int) -> int:
    return math.floor(val * precision)

def _to_state(board: PodBoard, pod: PodState) -> Tuple[int,int,int,int,int]:
    vel = pod.vel.rotate(-pod.angle)

    check1 = (board.get_check(pod.nextCheckId) - pod.pos).rotate(-pod.angle)
    check2 = (board.get_check(pod.nextCheckId + 1) - pod.pos).rotate(-pod.angle)

    return (
        _discretize(vel.x / Constants.max_vel(), 10),
        _discretize(vel.y / Constants.max_vel(), 10),
        _discretize(check1.x / MAX_DIST, 20),
        _discretize(check1.y / MAX_DIST, 20),
        _discretize(check2.angle(), 4)
    )


class QController(Controller):
    """
    A Controller that uses Q-Learning to win the race. The state and action spaces are discretized
    so that the table is manageable.
    """
    def __init__(self, board: PodBoard, reward_func: Callable[[PodBoard, PodState, PodState], float]):
        super().__init__(board)
        self.ad = ActionDiscretizer()
        self.reward_func = reward_func
        self.q_table = {}

        self.mins = {'x': 999999, 'y': 999999, 'vx': 999999, 'vy': 999999, 'ang': 999999}
        self.maxs = {'x': -999999, 'y': -999999, 'vx': -999999, 'vy': -999999, 'ang': -999999}

    def __record_minmax(self, pod: PodState):
        self.mins['x'] = min(self.mins['x'], pod.pos.x)
        self.mins['y'] = min(self.mins['y'], pod.pos.y)
        self.mins['vx'] = min(self.mins['vx'], pod.vel.x)
        self.mins['vy'] = min(self.mins['vy'], pod.vel.y)
        self.mins['ang'] = min(self.mins['ang'], pod.angle)
        self.maxs['x'] = max(self.maxs['x'], pod.pos.x)
        self.maxs['y'] = max(self.maxs['y'], pod.pos.y)
        self.maxs['vx'] = max(self.maxs['vx'], pod.vel.x)
        self.maxs['vy'] = max(self.maxs['vy'], pod.vel.y)
        self.maxs['ang'] = max(self.maxs['ang'], pod.angle)
        
    def __get_q_values(self, pod: PodState) -> List[float]:
        state = _to_state(self.board, pod)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.ad.num_actions)
        return self.q_table[state]

    def play(self, pod: PodState) -> PlayOutput:
        action = np.argmax(self.__get_q_values(pod))
        return self.ad.action_to_output(action, pod.angle, pod.pos)

    def __do_train(self,
                   pod: PodState,
                   max_turns: int,
                   prob_rand_action: float,
                   learning_rate: float,
                   future_discount: float
                   ) -> float:
        max_reward = self.reward_func(self.board, pod, pod)
        cur_check = pod.nextCheckId
        self.__record_minmax(pod)

        # Episode is done when we've hit a new checkpoint, or exceeded the max turns
        while pod.turns < max_turns and cur_check == pod.nextCheckId:
            # Choose an action
            if random.random() < prob_rand_action:
                action = math.floor(random.random() * self.ad.num_actions)
            else:
                action = np.argmax(self.__get_q_values(pod))

            # Take the action and calculate the reward. Since the discretization of the state space is
            # rough, we repeat the action until we get to a new state
            play = self.ad.action_to_output(action, pod.angle, pod.pos)

            next_pod = self.board.step(pod, play)
            reward = self.reward_func(self.board, pod, next_pod)
            self.__record_minmax(next_pod)

            # Update the Q-table
            cur_state_q = self.__get_q_values(pod)
            next_state_q = self.__get_q_values(next_pod)
            cur_state_q[action] = \
                (1 - learning_rate) * cur_state_q[action] + \
                learning_rate * (reward + future_discount * max(next_state_q))

            max_reward = max(reward, max_reward)
            pod = next_pod

        return max_reward

    def train(self,
              num_episodes: int = 10,
              max_turns: int = 50,
              prob_rand_action: float = 0.5,
              learning_rate: float = 0.5,
              future_discount: float = 0.8
              ) -> List[float]:
        """
        Train starting at a fixed faraway point
        """
        max_reward_per_ep = []

        for episode in range(num_episodes):
            # The pod starts in a random position at a fixed (far) distance from the check,
            # pointing in a random direction
            pos_offset = UNIT.rotate(random.random() * 2 * math.pi) * Constants.check_radius() * 17
            pod = PodState(
                pos=self.board.checkpoints[0] + pos_offset,
                angle=2 * math.pi * random.random() - math.pi
            )

            max_reward_per_ep.append(self.__do_train(
                pod,
                max_turns,
                prob_rand_action,
                learning_rate,
                future_discount))

        return max_reward_per_ep

    def train_iteratively(
            self,
            pods: List[PodState],
            max_turns: int = 50,
            prob_rand_action: float = 0.5,
            learning_rate: float = 0.5,
            future_discount: float = 0.8
    ) -> List[float]:
        """
        Train by iterating through all given states
        """
        max_reward_per_ep = []

        for pod in pods:
            max_reward_per_ep.append(self.__do_train(
                pod.clone(),
                max_turns,
                prob_rand_action,
                learning_rate,
                future_discount))

        return max_reward_per_ep
