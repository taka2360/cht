from collections import namedtuple, deque

Transition = namedtuple('Transition', ('sentence', 'nex_word', 'reward'))

class ReplayMemory:
    """学習に利用する（状態, 行動, 次状態, 報酬）を記憶"""
    def __init__(self):
        self.memory = deque([])

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)
