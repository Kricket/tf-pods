import math


class Constants:
    @staticmethod
    def world_x():
        return 16000
    @staticmethod
    def world_y():
        return 9000
    @staticmethod
    def border_padding():
        return 1000
    @staticmethod
    def check_spacing():
        return 3000
    @staticmethod
    def min_checks():
        return 3
    @staticmethod
    def max_checks():
        return 6
    @staticmethod
    def check_radius():
        return 600
    @staticmethod
    def check_radius_sq():
        return Constants.check_radius() * Constants.check_radius()
    @staticmethod
    def pod_radius():
        return 400
    @staticmethod
    def max_turn():
        return math.radians(18.)
    @staticmethod
    def friction():
        return 0.85
    @staticmethod
    def max_thrust():
        return 100
    @staticmethod
    def max_vel():
        """
        Maximum speed that a pod can attain through normal acceleration (tested empirically)
        """
        return 558
