import numpy as np

###CONSTANTS FOR TESTING###
# Uncomment np.random.seed(seed) when testing
seed = 42


class Env:
    def __init__(self, Sn, An, reward):
        
        if type(Sn) != int:
          raise TypeError("The number of states must be an integer")
        self.state_space = self.make_state_space(Sn)
        
        if type(An) != int:
          raise TypeError("The number of actions must be an integer")
        self.action_space = self.make_action_space(An)
        
        self.reward_per_step = reward
        

    def make_state_space(self, Sn: int):
        return ['S'+str(i) for i in range(Sn)]

    def make_action_space(self, An: int):
        return ['A'+str(i) for i in range(An)]

    def make_reward_func(self):
        reward = np.full((len(self.state_space), ), self.reward_per_step)
        reward[-1] = 0  # change the reward for the terminal state to 0
        return reward

    # TODO: Refactor get_number
    @staticmethod
    def get_number(state_or_act):
        list_char = [char for char in state_or_act]
        num = [char for char in list_char if char in {'0', '1', '2', '3', '4',
                                                      '5', '6', '7', '8', '9'}]
        num = int(''.join(num))
        return num

    #TODO: refactor list[0], by using generator?
    def reset(self):
        
        initial_state = (np.random.choice(self.state_space, 1))[0]
        while initial_state == self.state_space[-1]:
            initial_state = (np.random.choice(self.state_space, 1))[0]
        return initial_state

    def sample_action(self, policy, state):
        state_num = self.get_number(state)
        return np.random.choice(self.action_space, p=policy[state_num])
        

    def step(self, current_state, action, action_matrices):
        # np.random.seed(42)
        act_num = self.get_number(action)
        state_num = self.get_number(current_state)
        terminal_state = self.state_space[-1]
        next_state = np.random.choice(self.state_space,
                                      p=action_matrices[act_num, state_num])
        done = False
        if next_state == terminal_state:
            done = True
        reward = self.reward_per_step

        output = [next_state, reward, done]

        return output

# TODO: Add Value Iteration for Q-function, Policy Iteration,
#       and Value Iteration algorithms for control problem


class Agent:
    def __init__(self, policy, action_matrices):
        self.policy = policy
        self.action_matrices = action_matrices

    def make_policy_transition_matrix(self):
        n_actions, Sn_row, Sn_col = (self.action_matrices).shape
        list_policy_transition_matrices = []
        for i in range(Sn_row):
            for j in range(Sn_col):
                P_ij = 0
                for action in range(n_actions):
                    P_ij += self.policy[i, action] * \
                        self.action_matrices[action, i, j]
                list_policy_transition_matrices.append(P_ij)

        policy_transition_matrix = np.array(
            list_policy_transition_matrices).reshape(Sn_row, Sn_col)

        return policy_transition_matrix

    def collect_an_episode(self, env, initial_state=None):

        # reset the env
        if initial_state is not None:
            state = initial_state
        else:
            state = env.reset()

        # Create 2 empty lists for logging states and rewards
        ep_states = []
        ep_rews = []

        # not in the terminal state flag
        done = False

        # start collecting an episode
        while not done:
            # append the current state
            ep_states.append(state)

            # sample an action based on the policy and the current state
            sample_action = env.sample_action(self.policy, state)

            # Take action, receive the next state, reward, and indication whether in the terminal state
            state, reward, done = env.step(
                state, sample_action, self.action_matrices)

            # Log the reward
            ep_rews.append(reward)

        return ep_states, ep_rews

    def value_iteration(self, n, reward_func, gamma):
        V_0 = np.zeros(len(reward_func))
        V_i = V_0
        for i in range(n):
            V_i = reward_func + gamma * \
                (self.make_policy_transition_matrix().dot(V_i))
        return V_i

    def make_greedy_policy(self, value_function, policy=None):
        if policy is None:
            policy = self.policy
        
        state_num, action_num = policy.shape

        for state in range(state_num-1):
            #log the values of the next states
            next_states_val = []
            #log the index of the next states
            next_states_id = []
        
            for action in range(action_num):
                next_states_prob = self.action_matrices[action][state]
                for i in range(len(next_states_prob)):
                    if next_states_prob[i] == 1:
                        next_states_val.append(value_function[i])
                        next_states_id.append(i)
            
            #Pick the state with highest value
            max_state_val = np.max(next_states_val)  
            
            #Make greedy policy
            greedy_actions = [action_i for action_i in range(len(next_states_val)) 
                                    if next_states_val[action_i] == max_state_val]
                
            policy[state] = np.zeros(4)
            for action in greedy_actions:
                policy[state][action] = 1
            policy[state] = policy[state]/len(greedy_actions)
        
        return policy