import random
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, values: Tuple):
        self.buffer.append(values)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def is_full(self):
        return len(self.buffer) >= self.capacity

    def sample(self, size: int):
        return random.sample(self.buffer, size)

    def clear(self):
        self.buffer = []