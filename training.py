import gym
import numpy as np
import pandas as pd
import random

import pdb

class QLearn:
    def __init__(self, alpha, gamma, epsilon, actions):
        self.q = {} # replace the matrix with a dictionary instead
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions # [0, 1]

    def getQ(self, next_state, action):
        return self.q.get((state, action), 0.0)

    def learn(self, state, action, reward, next_state):
        # get max[Q(next_state, all actions)]
        maxQ_next_state = max([self.getQ(next_state, a) for a in self.actions])

        old_val = self.q.get((state, action), None)
        if old_val == None: 
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * maxQ_next_state)

    def choose_action(self, state):
        Q_vals_this_state = [self.getQ(state, a) for a in self.actions]

        # randomness to get out of local minima
        # if random.random() < self.epsilon:
        #     for pos in range(len(Q_vals_this_state)):
        #         Q_vals_this_state[pos] += random.random()

        max_Q_val = max(Q_vals_this_state)
        max_indices = [i for i, Q_val in enumerate(Q_vals_this_state) if Q_val == max_Q_val]

        if len(max_indices) > 1:
            idx = random.randint(0, len(max_indices)-1)
        else:
            idx = 0


        return self.actions[idx]

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    steps_history = np.ndarray(0)

    alpha = 0.1
    gamma = 0.7
    epsilon = 0.05
    actions = range(env.action_space.n)

    qlearn = QLearn(alpha, gamma, epsilon, actions)

    n_episodes = 50
    n_goal_steps = 195
    n_bins = 8
    n_bins_angle = 10
    n_features = env.observation_space.shape[0]

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    for i_episode in range(n_episodes):
        observation = env.reset()

        cart_pos, pole_angle, cart_vel, angular_change = observation
        state = build_state([to_bin(cart_pos, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_vel, cart_velocity_bins),
                             to_bin(angular_change, angle_rate_bins)])

        for t in range(n_goal_steps):
            # env.render()
            print(observation)

            # select random initial state
            action = qlearn.choose_action(state)
            print(action)
            observation, reward, done, info = env.step(action) 

            cart_pos, pole_angle, cart_vel, angular_change = observation
            next_state = build_state([to_bin(cart_pos, cart_position_bins),
                                     to_bin(pole_angle, pole_angle_bins),
                                     to_bin(cart_vel, cart_velocity_bins),
                                     to_bin(angular_change, angle_rate_bins)])

            # failing to meet the number of goal steps
            if done:
                reward = -200 # update reward from 0 to become -200
                qlearn.learn(state, action, reward, next_state)
                steps_history = np.append(steps_history, [t+1]) # keep track of our learnign progress
                print("Episode finished after {} timesteps".format(t+1))
                break
            # game still in progress
            else:
                qlearn.learn(state, action, reward, next_state)
                state = next_state

    print("avg number of steps taken is: ", np.mean(steps_history))
    env.close()