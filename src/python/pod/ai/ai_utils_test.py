import math
from unittest import TestCase

from pod.ai.ai_utils import gen_pods
from pod.util import PodState
from vec2 import Vec2


class AIUtilsTest(TestCase):
    def test_gen_pods(self):
        check = Vec2(10, 20)
        pods = gen_pods(
            [check],
            # One to the left, one to the right of the check
            [0, math.pi],
            [1.0],
            # Always pointing at the check
            [0.0],
            # One to the left, one to the right of the pod's heading (so +/- y)
            [math.pi / 2, 3 * math.pi / 2],
            [2.0])

        for pod in [
            PodState(pos=check + Vec2(1, 0), vel=Vec2(0,  2), angle=math.pi, next_check_id=0),
            PodState(pos=check + Vec2(1, 0), vel=Vec2(0, -2), angle=math.pi, next_check_id=0),
            PodState(pos=check - Vec2(1, 0), vel=Vec2(0,  2), angle=0, next_check_id=0),
            PodState(pos=check - Vec2(1, 0), vel=Vec2(0, -2), angle=0, next_check_id=0)
        ]:
            self.assertIn(pod, pods, "{} not found in {}".format(pod, [str(p) for p in pods]))

        self.assertEqual(len(pods), 4)
