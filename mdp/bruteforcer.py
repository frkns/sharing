import numpy as np

n, m = (3, 4)

grid = [
        [0,0,0,0],  # 1:player 2:obstacle
        [0,2,0,0], 
        [0,0,0,0]
        ] 
grid = grid[::-1]  # Cartesian indexing* except row first i.e. (y, x)

rewards = [
        [0,0,0,1],  # terminates when a single non-zero reward is collected
        [0,0,0,-1], 
        [0,0,0,0]
        ]
rewards = rewards[::-1]

rewards = np.array(rewards, dtype='float')
rewards[rewards == 0] = -0.02  # add a penalty for walking around

# 0.8 for going in the selected direction 0.1 to slip to either adjacent one
intended_p = 0.8
slip_p = (1 - intended_p) / 2

actions = {0, 1, 2, 3}  # 0:up, 1:down, 2:right, 3:left
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

dy = [1, -1, 0, 0]
dx = [0, 0, 1, -1]

states = set()
for i in range(3):
    for j in range(4):
        if grid[i][j] != 2:
            states.add((i, j))

gamma = 0.99

transitions = {}  # mapping: (state, action) -> { next_state: probability }
for state in states:
    i, j = state
    for a in range(4):
        transitions[(state, a)] = {}

        # intended
        next_i = i + dy[a]
        next_j = j + dx[a]
        if next_i < 0 or next_i >= n or next_j < 0 or next_j >= m or grid[next_i][next_j] == 2:
            intended_state = (i, j)  # blocked → no-op
        else:
            intended_state = (next_i, next_j)
        transitions[(state, a)][intended_state] = transitions[(state, a)].get(intended_state, 0) + intended_p

        # slip
        if dy[a] == 0:  # horizontal action → slip vertically
            for slip in (-1, 1):
                slip_i, slip_j = i + slip, j
                if slip_i < 0 or slip_i >= n or slip_j < 0 or slip_j >= m or grid[slip_i][slip_j] == 2:
                    slip_state = (i, j)  # invalid slip → no-op
                else:
                    slip_state = (slip_i, slip_j)
                transitions[(state, a)][slip_state] = transitions[(state, a)].get(slip_state, 0) + slip_p
        else:  # vertical action → slip horizontally
            for slip in (-1, 1):
                slip_i, slip_j = i, j + slip
                if slip_i < 0 or slip_i >= n or slip_j < 0 or slip_j >= m or grid[slip_i][slip_j] == 2:
                    slip_state = (i, j)  # invalid slip → no-op
                else:
                    slip_state = (slip_i, slip_j)
                transitions[(state, a)][slip_state] = transitions[(state, a)].get(slip_state, 0) + slip_p

import numpy as np
np.random.seed(0)
rng = np.random.randint


pi_table = {state: rng(4) for state in states}

def pi(s):
    return pi_table[s]

def Ps(s):
    return { a: transitions[(s, a)] for a in range(4) }

def Psa(s, a):
    return Ps(s)[a]

def R(s):
    return rewards[s[0]][s[1]]


from sys import setrecursionlimit
setrecursionlimit(10000)
from functools import lru_cache

@lru_cache(maxsize=None)
def Vpi(s, mx_depth=50):  # works, but really could use vectorization
    if mx_depth <= 0:
        return 0
    v = R(s)
    if abs(v) == 1:  # terminate on absorbing states
        return v
    return v + sum(gamma * prob * Vpi(s_prime, mx_depth-1) for s_prime, prob in Psa(s, pi(s)).items())


def print_env():
    for ln in grid:
        lnout = ''
        for ch in ln:
            if ch == 0:
                lnout += '.'
            elif ch == 1:
                lnout += 'O'
            else:
                lnout += '#'
        print('\t' + lnout)

print("Environment:")
print_env()
print("Rewards grid:")
print(rewards[::-1])


def bruteforce_train():  # takes 2 hours to run on kaggle cpu
    print("\nBruteforcing 4^11 possible states")
    global pi_table
    best_Vsum = -2e9 
    best_pi_table = {}
    for bits in range(4**11):
        Vpi.cache_clear()
        base4 = np.base_repr(bits, base=4).zfill(11)
        arr = list(base4)
        pi_table = {state: int(arr[i]) for i, state, in enumerate(states)}
        Vpi_grid = np.zeros(rewards.shape)
        for (i,j) in states:
            Vpi_grid[i][j] = Vpi((i,j))
        # Vpi_grid = Vpi_grid[::-1]
        # print(Vpi_grid)
        Vsum = sum(Vpi_grid.flatten())
        if Vsum > best_Vsum:
            best_Vsum = Vsum
            best_pi_table = pi_table
        # print('Sum of V:', Vsum)
        # print('pi table:\n', pi_table)
        if bits % 1000 == 0:
            print(f'\n--- step {bits} ---')
            print('Best Sum of V', best_Vsum)
            print('Current Sum of V', Vsum)
            print('Current Base 4 Bits', base4)
            print('Current pi Table', pi_table)
    print('---\nBest Sum of V', best_Vsum)
    print('Best pi Table', best_pi_table)
    fout = ''
    fout += 'pi Table\n' + str(best_pi_table)
    fout += '\nachieving a Vsum of ' + str(best_Vsum)
    with open('mdpOutput.txt', 'a') as f:
        f.write('\n\nmdpBruteforcer output:\n')
        f.write(fout)


# bruteforce_train()

optimal_pi_table = {(0, 1): 3, (1, 2): 0, (2, 1): 2, (0, 0): 0, (0, 3): 3, (2, 0): 2, (2, 3): 0, (0, 2): 3, (2, 2): 2, (1, 0): 0, (1, 3): 0}  # result after finished training in 2 h
pi_table = optimal_pi_table


Vpi_grid = np.zeros(rewards.shape)
for (i,j) in states:
    Vpi_grid[i][j] = round(Vpi((i,j)), 2)
Vpi_grid = Vpi_grid[::-1]
print('\nAfter training V*:')
print(Vpi_grid)  # consistent with https://www.youtube.com/watch?v=d5gaWTo6kDM&t=905s

pi_grid = [[' ' for _ in range(m)] for _ in range(n)]
for (i,j) in states:
    if abs(rewards[i][j]) == 1:  # skip terminal states
        continue
    pi_grid[i][j] = (str(pi((i,j)))
        .replace(str(UP), '↑').replace(str(DOWN), '↓')
        .replace(str(LEFT), '←').replace(str(RIGHT), '→'))
pi_grid = pi_grid[::-1]
print('\nAfter training pi*:')
for r in pi_grid:
    print(str(r).replace("'", ''))


