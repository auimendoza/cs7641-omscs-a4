from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import gym
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def getEnv(envid):
  envnames = ['FrozenLake-v0', 'FrozenLake8x8-v0', 'FrozenLake16x16-custom']
  actions = ['<', 'v', '>', '^']
  actions2 = [u'\u2190',u'\u2193',u'\u2192',u'\u2191']
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
  
  return env, envname, P, R, actions, actions2

def runPolicy(env, episodes, policy):
  timesteps = []
  gtimesteps = []
  g = 0
  h = 0
  l = 0
  print("Running %d episodes..." % (episodes))
  for _ in range(episodes):
      #print("episode %d" % (episode))
      done = False
      observation = env.reset()
      rewards = 0
      for t in range(env._max_episode_steps):
          action = policy[observation]
          observation, reward, done, _ = env.step(action)
          rewards += reward
          if done:
              #print("Timesteps = %d rewards = %.3f" % (t+1, rewards))
              if env.env.desc[int(env.env.s/env.env.nrow), env.env.s % env.env.nrow] == b'G':
                #print("Goal reached.")
                g += 1
                gtimesteps.append(t+1)
              else:
                #print("Fell in hole.")
                h += 1
              timesteps.append(t+1)
              break
      if not done:
        timesteps.append(env._max_episode_steps)
        l += 1 
  print("goals/holes/lost: %d, %d, %d" % (g, h, l))

  return timesteps, gtimesteps, (g, h, l)

def byteslist2asciilist(b):    
  return list(map(lambda x: x.decode('utf-8'), b))

def getValMap(env):
  npMAP = np.apply_along_axis(byteslist2asciilist, 1, env.env.desc)
  MAP = np.apply_along_axis(''.join, 1, npMAP).tolist()
  mapnum = []
  for s in list(''.join(MAP)):
      if s == 'S':
          mapnum.append(8)
      if s == 'F':
          mapnum.append(4)
      if s == 'H':
          mapnum.append(1)
      if s == 'G':
          mapnum.append(10)
  return np.array(mapnum).reshape(16,16)

def getPolicyActions(policy, actions):
    return np.array([actions[i] for i in policy]).reshape(16,16)

def printNicePolicy(env, policy, actions, title, figname):
  mapnum = getValMap(env)
  policyactions = getPolicyActions(policy, actions)
  sns.heatmap(mapnum, annot=policyactions, fmt='', cbar=False, cmap="Blues")
  plt.yticks(rotation=0)
  plt.xlabel("")
  plt.ylabel("")
  plt.title(title)
  plt.gcf()
  plt.savefig(figname)
  plt.close()

def printPolicy(env, policy, actions):
  print(np.array([actions[i] for i in policy]).reshape(env.env.nrow, env.env.ncol))
