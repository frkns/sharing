from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self): 
        return len(self.buffer)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        length = len(self)
        if batch_size > length:
            print(f'WARNING: batch_size is greater than size of buffer, setting batch_size to {length}')
            batch_size = length
        states, actions, rewards, next_states, terminateds = zip(*random.sample(self.buffer, batch_size))  # without replacement
        return np.stack(states), np.array(actions), np.array(rewards), np.stack(next_states), np.array(terminateds)