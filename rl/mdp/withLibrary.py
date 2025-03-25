import numpy as np
import mdptoolbox

# Grid setup.
n, m = 3, 4
grid = [
    [0, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0]
]
grid = grid[::-1]  # Cartesian indexing.

# Rewards: non-terminals get -0.02; terminal cells have ±1.
rewards = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
]
rewards = np.array(rewards[::-1], dtype='float')
rewards[rewards == 0] = -0.02

# Dynamics: intended move 0.8; slip to either adjacent direction with 0.1.
intended_p = 0.8
slip_p = (1 - intended_p) / 2
actions = [0, 1, 2, 3]  # 0: up, 1: down, 2: right, 3: left
dy = [1, -1, 0, 0]
dx = [0, 0, 1, -1]

# Map non-obstacle states to indices.
states = []
state_idx = {}
for i in range(n):
    for j in range(m):
        if grid[i][j] != 2:
            state_idx[(i, j)] = len(states)
            states.append((i, j))
n_states = len(states)

def is_terminal(i, j):
    return abs(rewards[i, j]) == 1

# Build transition matrices P and reward matrix R_mat.
# P[a][s, s'] = probability of moving from state s to s' when taking action a.
P = [np.zeros((n_states, n_states)) for _ in actions]
R_mat = np.zeros((n_states, len(actions)))

for (i, j), s in state_idx.items():
    # For terminal states, we make them absorbing with no future reward.
    if is_terminal(i, j):
        for a in actions:
            P[a][s, s] = 1.0
            R_mat[s, a] = 0.0  # No additional reward once terminal.
        continue

    for a in actions:
        transitions = {}

        # Intended move.
        next_i = i + dy[a]
        next_j = j + dx[a]
        if next_i < 0 or next_i >= n or next_j < 0 or next_j >= m or grid[next_i][next_j] == 2:
            intended_state = (i, j)
        else:
            intended_state = (next_i, next_j)
        transitions[intended_state] = transitions.get(intended_state, 0) + intended_p

        # Slip moves.
        if dy[a] == 0:  # horizontal action → slip vertically.
            for slip in (-1, 1):
                slip_i, slip_j = i + slip, j
                if slip_i < 0 or slip_i >= n or slip_j < 0 or slip_j >= m or grid[slip_i][slip_j] == 2:
                    slip_state = (i, j)
                else:
                    slip_state = (slip_i, slip_j)
                transitions[slip_state] = transitions.get(slip_state, 0) + slip_p
        else:  # vertical action → slip horizontally.
            for slip in (-1, 1):
                slip_i, slip_j = i, j + slip
                if slip_i < 0 or slip_i >= n or slip_j < 0 or slip_j >= m or grid[slip_i][slip_j] == 2:
                    slip_state = (i, j)
                else:
                    slip_state = (slip_i, slip_j)
                transitions[slip_state] = transitions.get(slip_state, 0) + slip_p

        # Fill in transition and reward matrices.
        for next_state, prob in transitions.items():
            s_next = state_idx[next_state]
            P[a][s, s_next] += prob
            # The reward from a transition is given by the reward in the landing cell.
            R_mat[s, a] += prob * rewards[next_state[0], next_state[1]]

gamma = 0.99
vi = mdptoolbox.mdp.ValueIteration(P, R_mat, gamma)
vi.run()

# ... [after running value iteration with vi.run()] ...

# Convert vi.V (which is a tuple) into a list so we can modify it.
vi.V = list(vi.V)

# Post-process: assign terminal states their immediate reward.
for (i, j), s in state_idx.items():
    if is_terminal(i, j):
        vi.V[s] = rewards[i, j]

# Reassemble the computed values and policy into grid form.
V_grid = np.full((n, m), np.nan)
policy_grid = np.full((n, m), ' ')
arrow = {0: '↑', 1: '↓', 2: '→', 3: '←'}

for (i, j), s in state_idx.items():
    V_grid[i, j] = round(vi.V[s], 2)
    if not is_terminal(i, j):
        policy_grid[i, j] = arrow[vi.policy[s]]
    else:
        policy_grid[i, j] = 'T'  # Mark terminal.

print("Optimal Value Function:")
print(np.flipud(V_grid))
print("\nOptimal Policy:")
for row in np.flipud(policy_grid):
    print(' '.join(row))

