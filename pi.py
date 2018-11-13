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
discount = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
episodes = 1000

env, envname, P, R, actions = common.getEnv(envid)

print("* %s *" % (envname))
print("Policy Iteration")

ndiffs = []
ghls = []
ts = []
for d in discount:
  print("discount: %.1f" % (d))
  func = PolicyIteration(P, R, discount=d, max_iter=maxiter)
  func.run()
  print("best policy:")
  common.printPolicy(env, func.policy, actions)
  timesteps, ghl = common.runPolicy(env, episodes, func.policy)
  ndiffs.append(func.n_different)
  ghls.append(ghl)
  ts.append(np.mean(timesteps))

# plot goal hole or lost
for i in range(3):
  plt.plot(discount, np.array(ghls)[:, i], '.-')
plt.legend(['goal', 'hole', 'lost'], loc="best")
plt.ylabel("count")
plt.xlabel("discount")
plt.suptitle("Outcome after %d episodes" % (episodes))
plt.title(envname)
plt.gcf()
plt.savefig("%d-ghl.png" % (envid), bbox_inches="tight")
plt.close()

# plot timesteps
plt.plot(discount, ts, '.-')
plt.xlabel("discount")
plt.ylabel("mean episode timesteps")
plt.suptitle("mean episode timesteps vs discount parameter")
plt.title(envname)
plt.gcf()
plt.savefig("%d-ts.png" % (envid), bbox_inches="tight")
plt.close()

# plot convergence
maxlen = 0
for i in ndiffs:
    if len(i) > maxlen:
        maxlen = len(i)

for i in ndiffs:
    if len(i) < maxlen:
        i.extend([i[-1]]*(maxlen-len(i)))
    plt.plot(range(maxlen), i, '.-')
plt.legend(list(map(lambda x: "discount = %.1f" % (x), discount)), loc="best")
plt.ylabel('# of differing policies')
plt.xlabel('iterations')
plt.suptitle('Policy Iteration: Convergence')
plt.title(envname)
plt.gcf()
plt.savefig("%d-converge.png" % (envid), bbox_inches="tight")
plt.close()