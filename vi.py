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
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
episodes = 1000
epsilons = [1./100**(i+1) for i in range(5)]

env, envname, P, R, actions = common.getEnv(envid)

print("* %s *" % (envname))
print("Value Iteration")

eiters = []
eghls = []
ets = []
for epsilon in epsilons:
  iters = []
  ghls = []
  ts = []
  for discount in discounts:
    print("discount: %.1f, epsilon: %s" % (discount, str(epsilon)))
    func = ValueIteration(P, R, discount, max_iter=maxiter, epsilon=epsilon)
    func.run()
    print("best policy:")
    common.printPolicy(env, func.policy, actions)
    timesteps, ghl = common.runPolicy(env, episodes, func.policy)
    iters.append(func.iter)
    ghls.append(ghl)
    ts.append(np.mean(timesteps))
  eiters.append(iters)
  eghls.append(ghls)
  ets.append(ts)

# plot timesteps
for i in ets:
  plt.plot(discounts, i)
plt.legend(list(map(lambda x: "epsilon = %s" % (str(x)), epsilons)), loc="best")
plt.xlabel("discount")
plt.ylabel("mean episode timesteps")
plt.suptitle("mean episode timesteps")
plt.title(envname)
plt.gcf()
plt.savefig("%d-vi-ts.png" % (envid), bbox_inches="tight")
plt.close()

# plot convergence
for i in range(len(epsilons)):    
    plt.plot(discounts, eiters[i])
plt.legend(list(map(lambda x: "epsilon = %s" % (str(x)), epsilons)), loc="best")
plt.ylabel('iterations')
plt.xlabel('discount')
plt.suptitle('Value Iteration: Convergence')
plt.title(envname)
plt.gcf()
plt.savefig("%d-vi-converge.png" % (envid), bbox_inches="tight")
plt.close()

# plot goal hole or lost heatmap
goals = np.array(eghls)[:,:,0]
import seaborn as sns; sns.set()
ax = sns.heatmap(goals, annot=True, fmt='g', cmap="Oranges", linewidth=0.2, xticklabels=discounts, yticklabels=epsilons)
plt.yticks(rotation=0)
plt.xlabel("discount")
plt.ylabel("epsilon")
plt.title("Goals after %d episodes\n%s" % (episodes, envname))
plt.gcf()
plt.savefig("%d-vi-ghl.png" % (envid), bbox_inches="tight")
plt.close()
