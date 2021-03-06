import math
from random import random

EPSILON = 0.00001

class Vec2(object):
    def __init__(self, x, y):
        self._x = float(x)
        self._y = float(y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __str__(self):
        return "(%.3f, %.3f)" % (self.x, self.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self._x * other, self._y * other)
        elif isinstance(other, Vec2):
            # Dot product
            return self._x * other._x + self._y * other._y
        else:
            raise TypeError("Invalid multiplicative operand: " + type(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self._x + other, self._y + other)
        elif isinstance(other, Vec2):
            return Vec2(self._x + other._x, self._y + other._y)
        else:
            raise TypeError("Invalid additive operand: " + type(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self._x - other, self._y - other)
        elif isinstance(other, Vec2):
            return Vec2(self._x - other._x, self._y - other._y)
        else:
            raise TypeError("Invalid additive operand: " + type(other))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self._x / other, self._y / other)
        else:
            raise TypeError("Invalid additive operand: " + type(other))

    def __eq__(self, other):
        if not isinstance(other, Vec2):
            return False
        if math.fabs(self._x - other._x) > EPSILON:
            return False
        if math.fabs(self._y - other._y) > EPSILON:
            return False
        return True

    def length(self) -> float:
        return math.sqrt(self.square_length())

    def square_length(self) -> float:
        return (self._x * self._x) + (self._y * self._y)

    def normalize(self):
        sq_len = self.square_length()
        if sq_len < EPSILON:
            return Vec2(0, 0)

        length = math.sqrt(sq_len)
        return Vec2(self._x / length, self._y / length)

    def round(self):
        return Vec2(math.floor(self._x + 0.5), math.floor(self._y + 0.5))

    def truncate(self):
        return Vec2(math.floor(self._x), math.floor(self._y))

    def rotate(self, radians):
        cos = math.cos(radians)
        sin = math.sin(radians)
        return Vec2(cos * self._x - sin * self._y,
                    sin * self._x + cos * self._y)

    def angle(self) -> float:
        """
        Get the angle this vector is pointing in, in radians
        """
        if abs(self._x) < EPSILON:
            return math.pi / 2.0 if self._y > 0 else math.pi * 1.5

        if abs(self._y) < EPSILON:
            return 0.0 if self._x > 0 else math.pi

        atan = math.atan(self._y / self._x)
        if self._x < 0: atan += math.pi
        if atan < 0: atan += 2.0 * math.pi

        return atan

    @staticmethod
    def random(max_x: float, max_y: float) -> 'Vec2':
        return Vec2(random() * (max_x + 1), random() * (max_y + 1))

    def __hash__(self):
        return self._x.__hash__() + self._y.__hash__()


ORIGIN = Vec2(0, 0)
UNIT = Vec2(1, 0)
