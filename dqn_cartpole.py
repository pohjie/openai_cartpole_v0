import gym
import numpy as np
import pandas as pd

from keras import optimizers, Sequential
from keras.layers import Dense
from keras.models import Model

from collections import deque
import random
import pdb

class DQN:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, epsilon_decay, lr, actions, state_size):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.actions = actions # [0, 1]
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.q = self.initDQN()

    def initDQN(self):
        q = Sequential()
        q.add(Dense(16, input_dim=self.state_size, activation='relu'))
        q.add(Dense(16, activation='relu'))
        q.add(Dense(2, activation='linear')) # Output layer

        optimizer=optimizers.Adam(lr=self.lr)
        q.compile(loss='mse', optimizer=optimizer)

        return q

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = -20
            else:
                target = reward + self.gamma * np.amax(self.q.predict(next_state))

            target_f = self.q.predict(state)[0]
            target_f[action] = target
            target_f = np.reshape(target_f, (1, -1))
            
            # pdb.set_trace()
            self.q.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def choose_action(self, state):
        # exploration
        if random.random() < self.epsilon:
            return env.action_space.sample()

        q_val = self.q.predict(state)[0]

        return np.argmax(q_val)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    steps_history = np.ndarray(0)

    dqn = DQN(alpha=0.5, gamma=0.9, epsilon=1, epsilon_min=0.01, 
             epsilon_decay=0.995, lr=0.001, actions=range(env.action_space.n),
             state_size=4)

    n_episodes = 500
    n_goal_steps = 500
    n_features = env.observation_space.shape[0]

    for i_episode in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, 4])

        for t in range(n_goal_steps):
            # env.render()
            print(state)

            # select random initial state
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action) # observation is next_state
            next_state = np.reshape(next_state, [1, 4])

            dqn.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                steps_history = np.append(steps_history, [t+1])
                break

        if len(dqn.memory) > 31:
            dqn.learn(batch_size=32)

    # Observe training results
    print("avg number of steps taken is: ", np.mean(steps_history))

    steps_history[::-1].sort()
    print(steps_history)

    env.close()