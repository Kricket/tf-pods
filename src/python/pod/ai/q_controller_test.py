from unittest import TestCase

from pod.ai.q_controller import get_index, pod_to_state, TOTAL_STATES, DIST_SQ_STATES, ANG_STATES, \
    VEL_ANG_STATES, VEL_MAG_SQ_STATES
from pod.board import PodBoard
from pod.util import PodState
from vec2 import Vec2


class QControllerTest(TestCase):
    def test_get_index_value_small(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(0, get_index(1, table))

    def test_get_index_value_big(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(4, get_index(15, table))

    def test_get_index_value_just_before_last(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(4, get_index(8.9, table))

    def test_get_index_value_mid(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(2, get_index(7, table))

    def test_get_index_value_approx_lower(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(2, get_index(7.1, table))

    def test_get_index_value_approx_higher(self):
        table = [x for x in range(5, 10)]
        self.assertEqual(2, get_index(6.9, table))

    def do_get_index(self, table):
        for (idx, dq) in enumerate(table):
            found = get_index(dq, table)
            self.assertEqual(idx, found)

    def test_get_index_DIST_SQ(self):
        self.do_get_index(DIST_SQ_STATES)

    def test_get_index_ANG(self):
        self.do_get_index(ANG_STATES)

    def test_get_index_VEL_ANG(self):
        self.do_get_index(VEL_ANG_STATES)

    def test_get_index_VEL_MAG_SQ(self):
        self.do_get_index(VEL_MAG_SQ_STATES)

    def test_pod_to_state_works(self):
        board = PodBoard([Vec2(5000, 5000)])
        pod = PodState()
        state = pod_to_state(pod, board)
        self.assertLess(state, TOTAL_STATES)
