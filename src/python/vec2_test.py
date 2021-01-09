import math
from unittest import TestCase

from vec2 import Vec2


class Vec2Test(TestCase):
    def test_init_works(self):
        v = Vec2(1.2, 3.4)
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def test_mul_scalar_works(self):
        v1 = Vec2(1, 2)
        v2 = v1 * 3
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 2)
        self.assertEqual(v2.x, 3)
        self.assertEqual(v2.y, 6)

    def test_rmul_scalar_works(self):
        v1 = Vec2(1, 2)
        v2 = 3 * v1
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 2)
        self.assertEqual(v2.x, 3)
        self.assertEqual(v2.y, 6)

    def test_dot_product_works(self):
        v1 = Vec2(1, 2)
        v2 = Vec2(3, 4)
        self.assertEqual(v1 * v2, 11)

    def test_add_scalar_works(self):
        v1 = Vec2(1, 2)
        v2 = v1 + 5
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 2)
        self.assertEqual(v2.x, 6)
        self.assertEqual(v2.y, 7)

    def test_radd_scalar_works(self):
        v1 = Vec2(1, 2)
        v2 = 5 + v1
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 2)
        self.assertEqual(v2.x, 6)
        self.assertEqual(v2.y, 7)

    def test_add_vec_works(self):
        v1 = Vec2(1, 2)
        v2 = Vec2(3, 4)
        add = v1 + v2
        self.assertEqual(add.x, 4)
        self.assertEqual(add.y, 6)

    def test_sub_vec_works(self):
        v1 = Vec2(1, 2)
        v2 = Vec2(3, 5)
        diff = v1 - v2
        self.assertEqual(diff.x, -2)
        self.assertEqual(diff.y, -3)

    def test_length_works(self):
        v = Vec2(3, -4)
        self.assertEqual(v.length(), 5)

    def test_square_length_works(self):
        v = Vec2(2, 3)
        self.assertEqual(v.square_length(), 13)

    def test_round_up_works(self):
        v = Vec2(1.5, 3.9)
        r = v.round()
        self.assertEqual(r.x, 2)
        self.assertEqual(r.y, 4)

    def test_round_down_works(self):
        v = Vec2(1.49, 3.001)
        r = v.round()
        self.assertEqual(r.x, 1)
        self.assertEqual(r.y, 3)

    def test_truncate_works(self):
        v = Vec2(2.0001, 4.9999)
        t = v.truncate()
        self.assertEqual(t.x, 2)
        self.assertEqual(t.y, 4)

    def test_rotate_works(self):
        v = Vec2(1, 1)
        r = v.rotate(math.pi)
        self.assertAlmostEqual(r.x, -1.0)
        self.assertAlmostEqual(r.y, -1.0)

    def test_angle_works(self):
        v = Vec2(123, -123)
        self.assertEqual(v.angle(), 1.75 * math.pi)
