# cartpole.py
import os
import gym
from cartpole_nn import Agent
from RL.utils.non_atari_utils import plotLearning  # Might have to modify to work with your folder name(s)
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[4], lr=0.001)
    scores, eps_history = [], []

    n_games = 500
    max_steps_per_episode = 500

    for i in range(n_games):
        score = 0
        done = False
        step = 0
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncate, info = env.step(action)

            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
            step += 1
            if step >= max_steps_per_episode:
                done = True

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i + 1 for i in range(n_games)]
    folder_path = 'plots/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = folder_path + 'cartpole.png'
    plotLearning(x, scores, eps_history, filename)

    env.close()
    env = gym.make('CartPole-v1', render_mode='human')

    for i in range(1000):
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncate, info = env.step(action)
            score += reward
            observation = observation_

        print(f'Viewing game {i + 1}, Score: {score}')

    env.close()
