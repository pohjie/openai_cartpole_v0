import gym
import numpy as np

class QLearn:
    def __init__(self, alpha, gamma, epsilon, actions):
        self.q = {} # replace the matrix with a dictionary instead
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions # [0, 1]

    def getQ(self, next_state, action):
        return q.get((state, action), 0.0)

    def learn(self, state, action, reward, next_state):
        # get max[Q(next_state, all actions)]
        maxQ_next_state = max([self.getQ(next_state, a) for a in self.actions])

        old_val = self.q.get((state, action), None)
        if old_val == None: 
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * maxQ_next_state)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    steps_history = np.empty(0)

    qlearn = QLearn(0.1, 0.7, 0.1, [0, 1])
    
    n_episodes = 100
    n_goal_steps = 200

    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(n_goal_steps):
            env.render()
            print(observation)

            # select random initial state
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action) 

            # failing to meet the number of goal steps
            if done:
                reward = -200 # update reward from 0 to become -200
                # qlearn.learn(state, action, reward, next_state)
                np.append(steps_history, t) # keep track of our learnign progress
                print("Episode finished after {} timesteps".format(t+1))
                break
            # game still in progress
            # else:
            #     qlearn.learn(state, action, reward, next_state)

    env.close()