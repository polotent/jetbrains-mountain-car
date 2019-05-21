import gym
import numpy as np

from keras import Sequential
from keras.layers import Dense

from collections import namedtuple
from collections import deque

Hyperparameters = namedtuple('Hyperarameters', ['episodes', 'steps', 'gamma', 'sigma', 'elite_frac', 'population_size'])
params = Hyperparameters(500, 1000, 1, 0.5, 0.2, 50)


class Car:
    def __init__(self, env, agent):
        self.env = env
        self.env.seed(100)
        np.random.seed(100)
        self.agent = agent(self.env)

    # cross entropy method
    def cem(self, print_interval=1):
        n_elite = int(params.population_size * params.elite_frac)
        scores_deque = deque(maxlen=100)
        scores = []
        best_weight = params.sigma * np.random.randn()

        for i in range(params.episodes):
            weights_pop = [best_weight + (params.sigma * np.random.randn(self.agent.get_weights_dim())) for i in range(params.population_size)]
            rewards = np.array([self.agent.train(weights) for weights in weights_pop])

            elite_indices = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_indices]
            best_weight = np.array(elite_weights).mean(axis=0)

            reward = self.agent.train(best_weight)
            scores_deque.append(reward)
            scores.append(reward)

            if i > 0 and i % print_interval == 0:
                print(f'Episode {i}\tAverage Score: {np.mean(scores_deque):.2f}')

            if np.mean(scores_deque) >= 90.0:
                print(
                    f'\nSolved in {i - 100 + 1:d} iterations!\tAverage Score: {np.mean(scores_deque):.2f}')
                break
        return scores

    def train_agent(self):
        self.cem()

    def test(self):
        state = self.env.reset()
        while True:
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            action = self.agent.choose_action(state)
            self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if done:
                break
        self.env.close()


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.actions_size = self.env.action_space.shape[0]
        self.hidden = 16
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(input_dim=self.state_size, units=self.hidden, activation='relu'))
        model.add(Dense(units=self.actions_size, activation='tanh'))
        return model

    def get_weights_dim(self):
        return (self.state_size + 1) * self.hidden + (self.hidden + 1) * self.actions_size

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state)[0]

    def train(self, weights):
        s_size, h_size, a_size = self.state_size, self.hidden, self.actions_size
        fc1_end = (s_size * h_size) + h_size
        fc1_W = weights[:s_size * h_size].reshape(s_size, h_size)
        fc1_b = weights[s_size * h_size:fc1_end]
        fc2_W = weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size)
        fc2_b = weights[fc1_end + (h_size * a_size):]
        self.model.set_weights([fc1_W, fc1_b, fc2_W, fc2_b])
        episode_return = 0
        state = self.env.reset()
        for step in range(params.steps):
            state = np.reshape(state, [1, self.state_size])
            action = self.get_action(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * params.gamma**step
            if done:
                break
        return episode_return


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    game = Car(env, Agent)
    print('Started training (Cross-Entropy)')
    game.train_agent()
    game.test()