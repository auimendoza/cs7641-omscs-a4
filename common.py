from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import gym

def getEnv(envid):
  envnames = ['FrozenLake-v0', 'FrozenLake8x8-v0', 'FrozenLake16x16-custom']
  actions = ['<', 'v', '>', '^']
  envname = envnames[envid]

  env = None
  if envid in [0,1]:
    env = gym.make(envname)
  if envid == 2:
    MAPS16X16 = [
            "SFFFFFFFFFFFFFFF",
            "FFFFFFFFFFHHFFFF",
            "FFFHFFFFFFFFFFHF",
            "FFFFFHFFHFFFFFHF",
            "FFFHFFFFFFFHFFFF",
            "FHHFFFHFFFFFFFFF",
            "FHFFHFHFFFFHFFFF",
            "FFFFFHFFFFHFFFFF",
            "FFFFFFFFFHFFFFFH",
            "FHFFFHFFFFFHFFFF",
            "FFFFFFFFFHHHFFFF",
            "FFFHFFFFFHFFFFFF",
            "FFFHFFFFFFFFFHFF",
            "FFFHFFFFHFFFFFFF",
            "FFFFFFFFFFFHHFFF",
            "FFFHFFFHFFFFFFFG"
        ]
    env = TimeLimit(FrozenLakeEnv(desc=MAPS16X16, map_name="16x16"), 
                    max_episode_seconds=None, 
                    max_episode_steps=2000)

  nS = env.env.nS
  nA = env.env.nA
  S = range(nS)
  A = range(nA)
  P = np.zeros((nA,nS,nS))
  R = np.zeros((nA,nS,nS))
  envP = env.env.P

  for s in S:
    for a in A:
      for i in envP[s][a]:
        P[a][s][i[1]] += i[0]
        R[a][s][i[1]] += i[2]
  
  return env, envname, P, R, actions

def printPolicy(env, policy, actions):
  print(np.array([actions[i] for i in policy]).reshape(env.env.nrow, env.env.ncol))

def runPolicy(env, episodes, policy):
  timesteps = []
  g = 0
  h = 0
  l = 0
  for episode in range(episodes):
      print("episode %d" % (episode))
      done = False
      observation = env.reset()
      rewards = 0
      for t in range(env._max_episode_steps):
          action = policy[observation]
          observation, reward, done, _ = env.step(action)
          rewards += reward
          if done:
              print("Timesteps = %d rewards = %.3f" % (t+1, rewards))
              if env.env.desc[int(env.env.s/env.env.nrow), env.env.s % env.env.nrow] == b'G':
                print("Goal reached.")
                g += 1
              else:
                print("Fell in hole.")
                h += 1
              timesteps.append(t+1)
              break
      if not done:
        timesteps.append(env._max_episode_steps)
        l += 1 
  return timesteps, (g, h, l)