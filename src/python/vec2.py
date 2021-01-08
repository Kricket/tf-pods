import math

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
            return Vec2(self.x * other, self.y * other)
        elif isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        else:
            raise TypeError("Invalid multiplicative operand: " + type(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self.x + other, self.y + other)
        elif isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Invalid additive operand: " + type(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.sqrt(self.square_length())

    def square_length(self) -> float:
        return (self.x * self.x) + (self.y * self.y)

    def normalize(self):
        sq_len = self.square_length()
        if sq_len < EPSILON:
            return Vec2(0, 0)

        length = math.sqrt(sq_len)
        return Vec2(self.x/length, self.y/length)

    def round(self):
        return Vec2(math.floor(self.x + 0.5), math.floor(self.y + 0.5))

    def truncate(self):
        return Vec2(math.floor(self.x), math.floor(self.y))

    def rotate(self, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Vec2(cos * self.x - sin * self.y,
                    sin * self.x + cos * self.y)

    def angle(self) -> float:
        """
        Get the angle this vector is pointing in, in radians
        """
        if abs(self.x) < EPSILON:
            return math.pi / 2.0 if self.y > 0 else math.pi * 1.5

        if abs(self.y) < EPSILON:
            return 0.0 if self.x > 0 else math.pi

        atan = math.atan(self.y / self.x)
        if self.x < 0: atan += math.pi
        if atan < 0: atan += 2.0 * math.pi

        return atan


ORIGIN = Vec2(0, 0)
UNIT = Vec2(1, 0)
