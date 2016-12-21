# -*- coding: utf-8 -*-
import pg
import gym
import numpy as np


GAMMA = 0.98 # discount rate

def get_episode_returns(episode_rewards,final_value,gamma=GAMMA): # compute the discounted total reward of an episode
    n = len(episode_rewards)
    episode_returns = [0. for i in range(n)]
    episode_returns[-1] = episode_rewards[-1] + gamma * final_value
    for i in range(n-2,-1,-1):
        episode_returns[i] = episode_rewards[i] + gamma* episode_returns[i+1]
    return episode_returns
assert get_episode_returns([1,0,0,4],1,0.5) == [1 + 0.5**3 * 4 + 0.5**4,0.5**2 * 4 + 0.5**3, 0.5 * 4 + 0.5**2, 4 + 0.5] # test

env = gym.make("CartPole-v1") # load the environment
ac = pg.PolicyNet(4,16,2) # input layer: 4 dim, hidden layer: 16 dim, output layer 2 dim


tmax = 1000
env.monitor.start('movies/cartpole-experiment-1',force=True) # recode some movies
logfile = open("log.txt","w") # learning curve. after training, you can visualize the curve by running vis.py
for k in range(1000):
    episode_rewards = []
    episode_actions = []
    episode_states = []
    state = env.reset()
    for epi in range(tmax):
        env.render()
        episode_states.append(state)
        action = ac.getAction(state)
        state, reward, terminal, info = env.step(np.argmax(action))
        episode_rewards.append(reward)
        episode_actions.append(action)
        if terminal or epi == tmax - 1:
            if terminal:
                final_value = 0.
            elif epi == tmax - 1:
                final_value = ac.getValue(state)[0]
            episode_returns = get_episode_returns(episode_rewards,final_value)
            ac.trainNetworkV(episode_states,episode_returns) # training the value function (critic)
            ac.trainNetworkP(episode_states,episode_actions,episode_returns) # training the policy function (actor)
            eptr = sum(episode_rewards)
            print "episode:",k,"reward:",eptr
            logfile.write(str(eptr)+"\n")
            break
logfile.close()
env.monitor.close()
