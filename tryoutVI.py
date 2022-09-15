import numpy as np
import ValueIteration as VI

#####TRY OUT THE CODE FOR THE GRIDWORLD EXAMPLE#####

# Set state and action space dimesions and reward per step
Sn = 15
An = 4
reward = -1

# Set a random policy
policy = np.ones((15, 4))/4
policy[-1] = np.array([0, 0, 0, 1])
print(policy)

# Initialize action transition probability matrices
# for moving up, down, left, and right actions
# TODO: refactor action matrices

up_matrix = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        if i == 3 and j == 14:
            up_matrix[i][j] = 1
        elif i == 14 and j == 14:
            up_matrix[i][j] = 1
        elif (i == 0 or i == 1 or i == 2) and i == j:
            up_matrix[i][j] = 1
        elif not(i == 0 or i == 1 or i == 2 or i == 3 or i == 14) and (i-j == 4):
            up_matrix[i][j] = 1


print(up_matrix)

down_matrix = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        if i == 10 and j == 14:
            down_matrix[i][j] = 1
        elif i == 14 and j == 14:
            down_matrix[i][j] = 1
        elif (i == 11 or i == 12 or i == 13) and i == j:
            down_matrix[i][j] = 1
        elif not(i == 10 or i == 11 or i == 12 or i == 13 or i == 14) and (j-i == 4):
            down_matrix[i][j] = 1

print(down_matrix)

left_matrix = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        if i == 0 and j == 14:
            left_matrix[i][j] = 1
        elif i == 14 and j == 14:
            left_matrix[i][j] = 1
        elif (i == 3 or i == 7 or i == 11) and i == j:
            left_matrix[i][j] = 1
        elif not(i == 0 or i == 3 or i == 7 or i == 11 or i == 14) and (i-j == 1):
            left_matrix[i][j] = 1

print(left_matrix)

right_matrix = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        if i == 13 and j == 14:
            right_matrix[i][j] = 1
        elif i == 14 and j == 14:
            right_matrix[i][j] = 1
        elif (i == 2 or i == 6 or i == 10) and i == j:
            right_matrix[i][j] = 1
        elif not(i == 13 or i == 2 or i == 6 or i == 10 or i == 14) and (j-i == 1):
            right_matrix[i][j] = 1

print(right_matrix)

# Concatenate action matrices into a single nparray
action_matrices = np.array([up_matrix, down_matrix,
                            left_matrix, right_matrix])

# Initialize Env and Agent object
env = VI.Env(Sn, An, reward)
agent = VI.Agent(policy, action_matrices)

# Explore the env
print(f"The evironment state space is {env.state_space}\
    \nThe evironment action space is {env.action_space}\
    \nThe evironment reward per step is {env.reward_per_step}")

# Compute the value function by the value iteration algo with 10 updates
for n in range(501):
    if n % 100 == 0:
        print(agent.value_iteration(n,
                                    reward_func=env.make_reward_func(),
                                    gamma=1))

# Try out agent method for collecting an episode
# and compute average reward and compare with the
# value iteration algorithm

# Set policy and action probability transition matrices

for i in range(Sn - 1):
    Si_rets = []
    for j in range(1000):
        _, ep_rews = agent.collect_an_episode(env, initial_state='S'+str(i))
        ep_rews_total = sum(ep_rews)
        Si_rets.append(ep_rews_total)
    print(sum(Si_rets)/1000)
