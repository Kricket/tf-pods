import math
from unittest import TestCase

from pod.ai.ai_utils import play_to_action, action_to_play, THRUST_VALUES, ANGLE_VALUES, THRUST_INC, ANGLE_INC, \
    action_to_output, state_to_vector, MAX_VEL, MAX_DIST
from pod.constants import Constants
from vec2 import Vec2


class AIUtilsTest(TestCase):
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

    def test_actions_produce_all_possible_combinations(self):
        outputs = set()
        angles = set()
        thrusts = set()
        for action in range(0, THRUST_VALUES * ANGLE_VALUES):
            thrust, angle = action_to_play(action)
            outputs.add((thrust, angle))
            angles.add(angle)
            thrusts.add(thrust)

        self.assertEqual(len(outputs), THRUST_VALUES * ANGLE_VALUES)
        self.assertEqual(len(angles), ANGLE_VALUES)
        self.assertEqual(len(thrusts), THRUST_VALUES)

        for t in range(0, THRUST_VALUES):
            self.assertTrue((THRUST_INC * t) in thrusts)
        for a in range(0, ANGLE_VALUES):
            self.assertTrue((ANGLE_INC * a - Constants.max_turn()) in angles)

    def test_action_to_output_simple(self):
        action = play_to_action(100, 0)
        po = action_to_output(action, 0, Vec2(100, 100))
        # The thrust should not have changed
        self.assertEqual(po.thrust, 100)
        # The pod is at (100, 100), angle 0, requested turn 0...so it should be aiming in the x direction
        self.assertAlmostEqual(po.target.y, 100)
        self.assertGreater(po.target.x, 100)

    def test_action_to_output_turn_right(self):
        action = play_to_action(50, Constants.max_turn())
        pod_pos = Vec2(100, 100)
        po = action_to_output(action, 1.23, pod_pos)
        # The thrust should not have changed
        self.assertEqual(po.thrust, 50)
        # The pod is at (100, 100), angle 1.23, requested turn max_turn...
        # If we undo the move and rotate, we should have a vector down the X-axis (i.e. angle 0)
        rel_target = (po.target - pod_pos).rotate(-1.23 - Constants.max_turn())
        self.assertAlmostEqual(rel_target.y, 0)
        self.assertGreater(rel_target.x, 1)

    def test_action_to_output_turn_left(self):
        action = play_to_action(50, -Constants.max_turn())
        pod_pos = Vec2(100, 100)
        po = action_to_output(action, 1.23, pod_pos)
        # The thrust should not have changed
        self.assertEqual(po.thrust, 50)
        # The pod is at (100, 100), angle 1.23, requested turn max_turn...
        # If we undo the move and rotate, we should have a vector down the X-axis (i.e. angle 0)
        rel_target = (po.target - pod_pos).rotate(-1.23 + Constants.max_turn())
        self.assertAlmostEqual(rel_target.y, 0)
        self.assertGreater(rel_target.x, 1)

    def test_state_to_vector_works(self):
        state = state_to_vector(
            Vec2(100, 100), # position
            Vec2(0, MAX_VEL), # velocity: straight up
            -math.pi, # angle: pointing down -X
            Vec2(100 + MAX_DIST, 100), # target check: behind the pod in X direction
            Vec2(100 + MAX_DIST, 100 + MAX_DIST) # next check: right turn from target check
        )

        self.assertAlmostEqual(state[0], -0.5 * math.pi, msg="angle from pod to velocity")
        self.assertAlmostEqual(state[1], math.pi, msg="angle from pod heading to target check")
        self.assertAlmostEqual(state[2], 0.5 * math.pi, msg="angle from (pod to check1) to (check1 to check2)")
        self.assertAlmostEqual(state[3], 1, msg="scaled velocity")
        self.assertAlmostEqual(state[4], 1, msg="scaled distance to check1")
        self.assertAlmostEqual(state[5], 1, msg="scaled distance check1 to check2")
