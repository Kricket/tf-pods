from unittest import TestCase

from pod.constants import Constants
from qnetwork.qpod_env import action_to_play, play_to_action


class EncodingTest(TestCase):
    def __do_p_a_p(self, thrust, angle):
        action = play_to_action(thrust, angle)
        new_thrust, new_angle = action_to_play(action)
        self.assertEqual(thrust, new_thrust)
        self.assertEqual(angle, new_angle)

    def test_play_to_action_to_play_0_angle(self):
        self.__do_p_a_p(20, 0)
    def test_play_to_action_to_play_min_angle(self):
        self.__do_p_a_p(30, -Constants.max_turn())
    def test_play_to_action_to_play_max_angle(self):
        self.__do_p_a_p(70, Constants.max_turn())
    def test_play_to_action_to_play_min_thrust(self):
        self.__do_p_a_p(0, 0)
    def test_play_to_action_to_play_max_thrust(self):
        self.__do_p_a_p(Constants.max_thrust(), 0)
    def test_play_to_action_to_play_both_max(self):
        self.__do_p_a_p(Constants.max_thrust(), Constants.max_turn())
    def test_play_to_action_to_play_both_min(self):
        self.__do_p_a_p(0, -Constants.max_turn())

    def test_play_to_action_to_play_rounds_thrust(self):
        action = play_to_action(3, 0)
        thrust, angle = action_to_play(action)
        self.assertEqual(thrust, 0)
        self.assertEqual(angle, 0)
    def test_play_to_action_to_play_rounds_angle(self):
        action = play_to_action(0, Constants.max_turn() * 0.001)
        thrust, angle = action_to_play(action)
        self.assertEqual(thrust, 0)
        self.assertEqual(angle, 0)
