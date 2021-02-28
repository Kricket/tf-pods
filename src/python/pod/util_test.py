import math
from unittest import TestCase

from pod.util import within, clean_angle, PodState
from vec2 import Vec2


class UtilTest(TestCase):
    def test_within_doesnt_change_if_ok(self):
        self.assertEqual(1, within(1, 0, 2))
    def test_within_works_when_too_low(self):
        self.assertEqual(1, within(0, 1, 2))
    def test_within_works_when_too_high(self):
        self.assertEqual(2, within(3, 1, 2))

    def test_clean_angle_doesnt_change_if_ok(self):
        self.assertEqual(0, clean_angle(0))
    def test_clean_angle_rotates_if_too_small(self):
        self.assertEqual(-0.5 * math.pi, clean_angle(-4.5 * math.pi))
    def test_clean_angle_rotates_if_too_big(self):
        self.assertEqual(0.5 * math.pi, clean_angle(4.5 * math.pi))

    def test_PodState_deserializes_equal_to_serialized(self):
        pod = PodState()

        pod.angle = 1.23
        pod.pos = Vec2(34, 56)
        pod.laps = 69
        pod.vel = Vec2(88, 77)
        pod.nextCheckId = 37

        ser = pod.serialize()
        copy = PodState()
        copy.deserialize(ser)

        self.assertEqual(pod.angle, copy.angle)
        self.assertEqual(pod.pos, copy.pos)
        self.assertEqual(pod.laps, copy.laps)
        self.assertEqual(pod.vel, copy.vel)
        self.assertEqual(pod.nextCheckId, copy.nextCheckId)

    def test_PodState_equals_initial(self):
        p1 = PodState()
        p2 = PodState()
        self.assertEqual(p1, p2)

    def test_PodState_equals_not_initial(self):
        p1 = PodState(pos = Vec2(1, 2), angle = 1.23, vel = Vec2(3, 4), next_check_id = 5)
        p2 = PodState(pos = Vec2(1, 2), angle = 1.23, vel = Vec2(3, 4), next_check_id = 5)
        self.assertEqual(p1, p2)

    def test_PodState_equals_fails_initial(self):
        p1 = PodState(pos=Vec2(0, 1))
        p2 = PodState(pos=Vec2(1, 1))
        self.assertNotEqual(p1, p2)
