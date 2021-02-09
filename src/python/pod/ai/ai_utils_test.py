import math
from unittest import TestCase

from pod.ai.ai_utils import gen_pods
from vec2 import Vec2, ORIGIN, UNIT


class AIUtilsTest(TestCase):
    def test_gen_pods(self):
        pods = gen_pods(
            ORIGIN,
            [1],
            [1.0],
            [0.0],
            [1.0])

        self.assertEqual(len(pods), 1)

        pod = pods[0]
        self.assertEqual(pod.pos, Vec2(1, 1))
        self.assertAlmostEqual(pod.angle, -.75 * math.pi + 1)

        expected_vel = UNIT.rotate(-.75 * math.pi)
        self.assertEqual(pod.vel, expected_vel, "{} should be {}".format(pod.vel,expected_vel))
