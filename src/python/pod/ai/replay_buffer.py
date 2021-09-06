import random
from typing import Tuple, List


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, values: Tuple):
        buf_len = len(self.buffer)
        if buf_len >= self.capacity:
            self.buffer[int(random.random() * buf_len)] = values
        else:
            self.buffer.append(values)

    def is_full(self):
        return len(self.buffer) >= self.capacity

    def sample(self, size: int) -> List[Tuple]:
        return random.sample(self.buffer, size)

    def clear(self):
        self.buffer = []