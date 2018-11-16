#!/Users/auimendoza/anaconda/envs/py36/bin/python

import gym
import numpy as np
from mdptoolbox.mdp import ValueIteration
from mdplib import PolicyIteration, QLearning
import common
import sys
import pandas as pd
import matplotlib.pyplot as plt

def Usage():
  print("Usage:")
  print("./%s <envid>" % (sys.argv[0]))
  sys.exit(1)

if len(sys.argv) < 2:
  Usage()

envid = int(sys.argv[1])

maxiter = 10000
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
episodes = 1000
epsilons = [1./100**(i+1) for i in range(5)]

env, envname, P, R, actions, actions2 = common.getEnv(envid)

print("* %s *" % (envname))
print("Value Iteration")

eiters = []
eghls = []
ets = []
bestgoal = 0
bestpolicy = None
bestpolicyparams = {}
for epsilon in epsilons:
  iters = []
  ghls = []
  ts = []
  for gamma in gammas:
    print("gamma: %.1f, epsilon: %s" % (gamma, str(epsilon)))
    func = ValueIteration(P, R, gamma, max_iter=maxiter, epsilon=epsilon)
    func.run()
    #print("best policy:")
    #common.printPolicy(env, func.policy, actions)
    timesteps, gtimesteps, ghl = common.runPolicy(env, episodes, func.policy)
    if ghl[0] > bestgoal:
      bestgoal = ghl[0]
      bestpolicy = func.policy
      bestpolicyparams['gamma'] = gamma
      bestpolicyparams['epsilon'] = epsilon
      bestpolicyparams['iterations'] = func.iter
      bestpolicyparams['elapsedtime'] = func.time
      bestpolicyparams['meangtimesteps'] = np.mean(gtimesteps)
    iters.append(func.iter)
    ghls.append(ghl)
    ts.append(np.mean(timesteps))
  eiters.append(iters)
  eghls.append(ghls)
  ets.append(ts)

# plot best policy
if envid == 0:
  textsize = 20
  textx = 4.25
  texty = 1
if envid == 2:
  textsize = 12
  textx = 16.5
  texty = 4
bestparamstext = "gamma=%.1f\nepsilon=%s\niterations=%d\nelapsed time=%.3f\nmean goal timesteps=%.3f" % (bestpolicyparams['gamma'],
    bestpolicyparams['epsilon'],
    bestpolicyparams['iterations'],
    bestpolicyparams['elapsedtime'],
    bestpolicyparams['meangtimesteps'])
common.printNicePolicy(env, bestpolicy, actions2, textsize, textx, texty, "Value Iteration: Best Policy\n%s" % (envname), bestparamstext, "%d-vi-bestpolicy.png" % (envid))

# plot timesteps
for i in ets:
  plt.plot(gammas, i)
plt.legend(list(map(lambda x: "epsilon = %s" % (str(x)), epsilons)), loc="best")
plt.xlabel("gamma")
plt.ylabel("mean episode timesteps")
plt.suptitle("Mean Episode Timesteps")
plt.title(envname)
plt.gcf()
plt.savefig("%d-vi-ts.png" % (envid), bbox_inches="tight")
plt.close()

# plot convergence
for i in range(len(epsilons)):    
    plt.plot(gammas, eiters[i])
plt.legend(list(map(lambda x: "epsilon = %s" % (str(x)), epsilons)), loc="best")
plt.ylabel('iterations')
plt.xlabel('gamma')
plt.suptitle('Value Iteration: Convergence')
plt.title(envname)
plt.gcf()
plt.savefig("%d-vi-converge.png" % (envid), bbox_inches="tight")
plt.close()

# plot goal hole or lost heatmap
goals = np.array(eghls)[:,:,0]
import seaborn as sns; sns.set()
ax = sns.heatmap(goals, annot=True, fmt='g', cmap="Oranges", linewidth=0.2, xticklabels=gammas, yticklabels=epsilons)
plt.yticks(rotation=0)
plt.xlabel("gamma")
plt.ylabel("epsilon")
plt.title("Goals After %d Episodes\n%s" % (episodes, envname))
plt.gcf()
plt.savefig("%d-vi-ghl.png" % (envid), bbox_inches="tight")
plt.close()
