# Uniform Experience Replay Memory & Proportional Prioritized Replay Memory

import random
import numpy as np
from sum_tree import SumTree


# Uniform Experience Replay Memory
class ExperienceReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def current_length(self):
        return len(self.memory)

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_sz):
        samples = random.sample(self.memory, batch_sz)
        return map(np.array, zip(*samples))


# Proportional Prioritized Experience Replay Memory
class PrioritizedReplayMemory:

    def __init__(self, capacity, alpha=0.6, eps=1e-2):
        self.tree = SumTree(capacity)
        self.alpha = alpha # alpha determines how much prioritization is used
        self.eps = eps # epsilon smooths priority, priority = (TD_error + eps) ** alpha

    def _get_priority(self, td_error):
        return (td_error + self.eps) ** self.alpha

    def current_length(self):
        return self.tree.current_length()

    def total_sum(self):
        return self.tree.total_sum()

    def push(self, event, td_error):
        priority = self._get_priority(td_error)
        self.tree.insert(event, priority)

    def sample(self, batch_sz):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_sum() / batch_sz

        for i in range(batch_sz):
            l = segment * i
            r = segment * (i + 1)

            s = random.uniform(l, r)
            (idx, priority, data) = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        samples = map(np.array, zip(*batch))

        return samples, indices, priorities

    def update(self, idx, td_error):
        if isinstance(idx, list):
            for i in range(len(idx)):
                priority = self._get_priority(td_error[i])
                self.tree.update(idx[i], priority)
        else:
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)

