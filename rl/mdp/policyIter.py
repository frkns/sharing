import numpy as np
np.random.seed(0)
rng = np.random.randint

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

V_table = {state: 0 for state in states}

def V(s):
    return V_table[s]

def Ps(s):
    return { a: transitions[(s, a)] for a in range(4) }

def Psa(s, a):  # -> (s_next, prob)
    return Ps(s)[a]

def R(s):
    return rewards[s[0]][s[1]]


def policy_iteration():
    policy = {s: UP for s in states}  # Initial policy: all UP
    state_list = sorted(states)  # Ordered list for matrix indexing
    state_index = {s: i for i, s in enumerate(state_list)}
    
    for iteration in range(1001):
        # === Policy Evaluation Phase ===
        # Construct linear system (I - γP)V = R
        size = len(state_list)
        A = np.eye(size)  # Identity matrix
        b = np.zeros(size)
        
        for i, s in enumerate(state_list):
            if abs(R(s)) >= 1:  # Terminal state
                A[i, i] = 1.0
                b[i] = R(s)
                continue
                
            a = policy[s]  # Current policy action
            b[i] = R(s)  # Immediate reward
            
            # Subtract γ * transition probabilities
            for s_prime, prob in Psa(s, a).items():
                j = state_index[s_prime]
                A[i, j] -= gamma * prob
        
        # Solve linear system
        try:
            V_values = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            V_values = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Update value table
        for s, v in zip(state_list, V_values):
            V_table[s] = v
        
        # === Policy Improvement Phase ===
        policy_stable = True
        for s in states:
            if abs(R(s)) >= 1:  # Skip terminals
                continue
                
            # Find best action
            best_a = None
            best_q = -np.inf
            for a in actions:
                q = sum(prob * V_table[s_p] for s_p, prob in Psa(s, a).items())
                q = R(s) + gamma * q
                if q > best_q:
                    best_q = q
                    best_a = a
            
            if best_a != policy[s]:
                policy[s] = best_a
                policy_stable = False
        
        print(f"Iteration {iteration+1}: Policy stable? {policy_stable}")
        if policy_stable:
            break
    
    return policy



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

policy = policy_iteration()

Vpi_grid = np.zeros(rewards.shape)
for (i,j) in states:
    Vpi_grid[i][j] = round(V((i,j)), 2)
Vpi_grid = Vpi_grid[::-1]
print('\nAfter training V*:')
print(Vpi_grid)  # consistent with bruteforcer

pi_grid = [[' ' for _ in range(m)] for _ in range(n)]
for (i,j) in states:
    if abs(rewards[i][j]) == 1:  # skip terminal states
        continue
    a = policy[(i,j)]
    pi_grid[i][j] = (str(a)
        .replace(str(UP), '↑').replace(str(DOWN), '↓')
        .replace(str(LEFT), '←').replace(str(RIGHT), '→'))
pi_grid = pi_grid[::-1]
print('\nAfter training pi*:')
for r in pi_grid:
    print(str(r).replace("'", ''))
