import pytest
import unittest
from pytest import approx
from ValueIteration import Env, Agent
import numpy as np


##CONSTANTS FOR TESTING RANDOM FUNCTION"
uncomment_seed = True

##CONSTANTS FOR THE ENV CLASS##

# Set the numpy random seed, number of states and actions, reward per step
seed = 42
Sn = 4
An = 4
reward = -1

# Initialize action transition probability matrices
action_matrices = (np.ones(64).reshape(4, 4, 4))/4
# Replace the last row for the terminal state
action_matrices[:, -1, :] = np.array([0, 0, 0, 1])

env = Env(Sn,
          An,
          reward,
          )

##CONSTANTS FOR THE AGENT CLASS##

# Initilize a random policy
policy = (np.ones(16).reshape(4, 4))/4

# Change the last row for the terminal state
policy[3] = np.array([0, 0, 0, 1])

agent = Agent(policy, action_matrices)


#################TEST ENV CLASS##########################
#### Test attributes of the Env class ####

def test_Sn_type():    
    with pytest.raises(TypeError) as excinfo:
        env = Env("Sn", 1, 2)
    assert  "The number of states must be an integer" == str(excinfo.value)

def test_state_space():
    assert env.state_space == ['S0', 'S1', 'S2', 'S3']

def test_An_type():    
    with pytest.raises(TypeError) as excinfo:
        env = Env(Sn, "An", 2)
    assert  "The number of actions must be an integer" == str(excinfo.value)

def test_action_space():
    assert env.action_space == ['A0', 'A1', 'A2', 'A3']


def test_reward_per_step():
    assert env.reward_per_step == reward


#### Test methods of the Env class ####
# TODO: Collect_an_episode method has not been tested

# Test attributes-related methods
def test_make_state_space():
    assert env.make_state_space(Sn) == ['S0', 'S1', 'S2', 'S3']


def test_make_action_space():
    assert env.make_action_space(An) == ['A0', 'A1', 'A2', 'A3']


def test_make_reward_func():
    # all_input = [env_ins.reward, env_ins.state_space]
    output = np.array([-1, -1, -1, 0])
    assert np.array_equal(env.make_reward_func(), output)


def test_reward_func_dim():
    assert (env.make_reward_func()).shape[0] == len(env.state_space)


def test_get_number():
    assert env.get_number('S1234') == 1234


# Test methods for the agent to interact with the environment under a policy

def test_reset_output_type():
    
    assert isinstance(env.reset(), str)    


def test_reset_nonterminal():
    initial_states = []
    for _ in range(1000):
        initial_states.append(env.reset())
    assert 'S4' not in initial_states


def test_sample_action_output_type():
    # all_inputs = [env.action_space, policy, 'S1']
    input = [policy, 'S1']
    assert isinstance(env.sample_action(*input), str)
  


@pytest.mark.skipif(uncomment_seed, reason="Uncomment seed in the main file for testing")
def test_step():
    np.random.seed(seed)
    # all_inputs = ['S2', 'A0', action_matrices, env.state_space]
    input = ['S2', 'A0', action_matrices]
    next_state = np.random.choice(env.state_space, p=action_matrices[0, 2])
    done = False
    output = [next_state, reward, done]
    assert env.step(*input) == output


#################TEST AGENT CLASS##########################
# TODO: Add Action-Value Iteration, Policy Iteration,
# Value Iteration for control algorithms

### Test agent attributes ###
def test_policy_shape():
    assert (agent.policy).shape == (len(env.action_space),
                                    len(env.state_space))


def test_policy_type():
    assert isinstance(policy, np.ndarray)


def test_action_matrices_shape():
    assert (agent.action_matrices.shape) == (len(env.action_space),
                                             len(env.state_space),
                                             len(env.state_space))


def test_action_matrices_normalized():
    normalized = agent.action_matrices.sum(axis=2)
    state_dim = len(env.state_space)
    assert np.array_equal(normalized, np.ones(
        state_dim**2).reshape(state_dim, state_dim))


### Test agent methods ###

# TODO: make this test more general. Now it can only test random policy


def test_make_policy_transition_matrix():
    output = action_matrices[0, :, :]
    assert np.array_equal(agent.make_policy_transition_matrix(), output)

# Value iteration algorithm

def test_value_iter():
    input = [10, env.make_reward_func(), 1]
    output = np.array([-3.77474594, -3.77474594, -3.77474594,  0.])    
    assert agent.value_iteration(*input) == approx(output)


@pytest.mark.skip(reason="Change this test case to test the random policy for Sn=4 and An=4")
def test_make_greedy_policy():
    # all_inputs = [value_function, policy, agent.action_matrices]
    value_function = np.array([-14, -20, -22, -14, -18, -20, -20, -20, 
                               -20, -18, -14, -22, -20, -14, 0], dtype = np.float32)
    policy = np.ones((15,4), dtype=np.float32)/4
    policy[-1] = [0., 0., 0., 1.0]
    input = [value_function, policy]
    output =  np.array([[0.,  0.,  1.,  0., ],
                        [0.,  0.,  1.,  0., ],
                        [0.,  0.5, 0.5, 0., ],
                        [1.,  0.,  0.,  0., ],
                        [0.5, 0.,  0.5, 0., ],
                        [0.,  0.5, 0.5, 0., ],
                        [0.,  1.,  0.,  0., ],
                        [1.,  0.,  0.,  0., ],
                        [0.5, 0.,  0.,  0.5,],
                        [0.,  0.5, 0.,  0.5,],
                        [0.,  1.,  0.,  0., ],
                        [0.5, 0.,  0.,  0.5,],
                        [0.,  0.,  0.,  1., ],
                        [0.,  0.,  0.,  1., ],
                        [0.,  0.,  0.,  1., ]])
    assert np.array_equal(agent.make_greedy_policy(*input), output)
