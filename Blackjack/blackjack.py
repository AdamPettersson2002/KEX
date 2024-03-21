# blackjack.py
import os
import gymnasium as gym
from blackjack_nn import Agent
from RL.utils.non_atari_utils import plotBlackjackLearning  # Might have to modify to work with your folder name(s)
import numpy as np

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[3], lr=0.001)
    scores, eps_history, wins, losses, draws = [], [], [], [], []
    win_count, loss_count, draw_count, total_games = 0, 0, 0, 0
    n_games = 50000

    for i in range(n_games):
        score = 0
        done = False
        step = 0
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncate, info = env.step(action)

            if reward == 1:
                win_count += 1
            elif reward == -1:
                loss_count += 1
            else:
                draw_count += 1
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
            step += 1

        total_games = i + 1
        wins.append(win_count / total_games)
        losses.append(loss_count / total_games)
        draws.append(draw_count / total_games)
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

    filename = folder_path + 'blackjack.png'
    plotBlackjackLearning(x, scores, wins, losses, draws, filename, epsilons=eps_history)

    env.close()
