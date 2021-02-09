import math
from unittest import TestCase

from pod.ai.ai_utils import MAX_DIST
from pod.ai.deep_reward_controller import state_to_vector, STATE_VECTOR_LEN
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState
from vec2 import Vec2, ORIGIN


class DeepRewardControllerTest(TestCase):
    def test_state_to_vector_works1(self):
        # A pod at (100, 100) pointing down -X, moving full speed +Y
        pod = PodState(Vec2(100, 100), Vec2(0, Constants.max_vel()), -math.pi)
        # The target checkpoint is directly behind it
        board = PodBoard([Vec2(100 + MAX_DIST, 100), ORIGIN])

        state = state_to_vector(pod, board)

        self.assertEqual(len(state), STATE_VECTOR_LEN)
        self.assertAlmostEqual(state[0], 0, msg="velocity x")
        self.assertAlmostEqual(state[1], -1, msg="velocity y")
        self.assertAlmostEqual(state[2], -1, msg="check1 x")
        self.assertAlmostEqual(state[3], 0, msg="check1 y")

    def test_state_to_vector_works2(self):
        # A pod at (-100, -100) pointing up +Y, moving 45 degrees down-left
        pod = PodState(Vec2(-100, -100), Vec2(-3, -3), math.pi / 2)
        # The target checkpoint is directly in front
        board = PodBoard([Vec2(-100, 1000), ORIGIN])

        state = state_to_vector(pod, board)

        self.assertEqual(len(state), STATE_VECTOR_LEN)
        self.assertAlmostEqual(state[0], -3 / Constants.max_vel(), msg="velocity x")
        self.assertAlmostEqual(state[1], 3 / Constants.max_vel(), msg="velocity y")
        self.assertAlmostEqual(state[2], 1100 / MAX_DIST, msg="check1 x")
        self.assertAlmostEqual(state[3], 0, msg="check1 y")
