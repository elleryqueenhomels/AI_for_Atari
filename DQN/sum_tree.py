# Sum-Tree: a binary tree data structure (array-based)
# where the parent's value is the sum of its children.

import numpy as np


class SumTree:

    def __init__(self, capacity):
        self.is_full = False
        self.write_index = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        # current node is leaf node
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def current_length(self):
        if self.is_full:
            return self.capacity
        else:
            return self.write_index

    def total_sum(self):
        return self.tree[0]

    def insert(self, data, priority):
        self.data[self.write_index] = data

        idx = self.write_index + self.capacity - 1
        self.update(idx, priority)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0
            self.is_full = True

    def update(self, idx, priority):
        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        # returns tuple: (tree_idx, priority, data)
        return (idx, self.tree[idx], self.data[data_idx])

