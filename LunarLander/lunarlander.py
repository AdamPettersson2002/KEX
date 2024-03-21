# lunarlander.py
import os
import gymnasium as gym
from lunarlander_nn import Agent
from RL.utils.non_atari_utils import plotLearning  # Might have to modify to work with your folder name(s)
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # env = TransformReward(env, )
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []

    n_games = 500
    max_steps_per_episode = 1000

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

    filename = folder_path + 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)

    env.close()
    env = gym.make('LunarLander-v2', render_mode='human')

    for i in range(1000):
        step = 0
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncate, info = env.step(action)
            score += reward
            observation = observation_
            step += 1
            if step >= 3*max_steps_per_episode:
                done = True

        print(f'Viewing game {i + 1}, Score: {score}')

    env.close()
