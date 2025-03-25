from gymnasium.wrappers import FrameStackObservation
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from pathlib import Path
import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import ale_py
import time
import cv2
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, ...)
                                              # input (m, 4,84,84) where m=batch_size
        self.conv1 = nn.Conv2d(4, 32, 8, 4)   # -> leaky_relu (m, 32,20,20)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)  # -> leaky_relu (m, 64,9,9)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)  # -> leaky_relu (m, 64,7,7) -> flatten (m, 3136)

        # nn.Linear(in_features, out_features, ...)
        self.fc1 = nn.Linear(3136, 512)  # -> leaky_relu (m, 512)
        self.fc2 = nn.Linear(512, 4)  # -> (m, 4) -> output
        
    def forward(self, x):  # expect uint8 tensor as input
        x = x / 255.0
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten it to (m, 3136)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)  # no output activation
        return x


class ReplayBuffer:  # we will store frames in uint8 to save memory. the forward pass of the network can convert it to float32 if desired
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self): 
        return len(self.buffer)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        assert batch_size <= len(self), 'sample size is greater than population of buffer'
        states, actions, rewards, next_states, terminateds = zip(*random.sample(self.buffer, batch_size))  # without replacement
        return torch.stack(states).squeeze(1), np.array(actions), np.array(rewards), torch.stack(next_states).squeeze(1), np.array(terminateds)

        
def get_checkpoint(v=-1, path='./checkpoints'):

    '''
    If found returns (file_name, version).
    Otherwise, returns (None, 0).
    '''
    ls = os.listdir(path) 
    if not ls: return (None, 0)
    mx = -1
    mx_file = ''
    for f in ls:
        try:
            cur = int(f.split('-')[0])  # might be a 'mem-...' file
        except:
            continue
        if cur > mx:
            mx = cur
            mx_file = f 
        if cur == v:
            return f, v
    return mx_file, mx
    

def to_grayscale(stack):  # expects a 4-frame stack
    # cv2 COLOR_BGR2GRAY is so fast!
    return np.array([cv2.cvtColor(stack[i], cv2.COLOR_BGR2GRAY) for i in range(stack.shape[0])])
    
def show_image(image_array):  # (84, 84)
    plt.imshow(Image.fromarray(image_array)), plt.axis('off'), plt.show()

def show_stack(stack):  # k-frame stack (4, 84, 84)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(Image.fromarray(stack[i]))
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def show_batch(batch):  # batch of 4-frame stacks of length m, shape (m, 4, 84, 84), displays first 4-frame stack
    show_stack(np.array(batch.detach()[0]) * 255)

def downsample_frame(f):
    f = f[31:-12:2, ::2]
    return np.pad(f, ((0, 0), (2, 2)), mode='constant')

def downsample_stack(stack):
    return np.array([downsample_frame(stack[i]) for i in range(stack.shape[0])])

def preprocess_state(s):  # 4-frame stack of shape (4, 84, 84)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = to_grayscale(s)
    s = downsample_stack(s)
    return torch.tensor(s, dtype=torch.uint8, device=device).unsqueeze(0)  # unsqueeze for the batch dim