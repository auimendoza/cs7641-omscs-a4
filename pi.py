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

mdpid = 'pi'
mdpname = 'Policy Iteration'
envid = int(sys.argv[1])

maxiter = 10000
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
episodes = 1000

env, envname, P, R, actions, actions2 = common.getEnv(envid)

print("* %s *" % (envname))
print(mdpname)

ndiffs = []
ghls = []
ts = []
bestgoal = 0
bestpolicy = None
bestpolicyparams = {}
print("Running ...")
for gamma in gammas:
  #print("gamma: %.1f" % (gamma))
  func = PolicyIteration(P, R, discount=gamma, max_iter=maxiter)
  func.run()
  #print("best policy:")
  #common.printPolicy(env, func.policy, actions)
  timesteps, gtimesteps, ghl = common.runPolicy(env, episodes, func.policy)
  if ghl[0] > bestgoal:
    bestgoal = ghl[0]
    bestpolicy = func.policy
    bestpolicyparams['gamma'] = gamma
    bestpolicyparams['iterations'] = func.iter
    bestpolicyparams['elapsedtime'] = func.time
    bestpolicyparams['meangtimesteps'] = np.mean(gtimesteps)
    bestpolicyparams['ndiff'] = func.n_different[-1]
  ndiffs.append(func.n_different)
  ghls.append(ghl)
  ts.append(np.mean(timesteps))

# plot best policy
textsize = 12
if envid == 0:
  textsize = 20
print("== best policy ==")
print("goals = %d" % (bestgoal))
print("gamma = %.1f" % (bestpolicyparams['gamma']))
print("differing policies = %d" % (bestpolicyparams['ndiff']))
print("iterations = %d" % (bestpolicyparams['iterations']))
print("elapsed time = %.3f" % (bestpolicyparams['elapsedtime']))
print("mean timesteps to goal = %.3f" % (bestpolicyparams['meangtimesteps']))
print("=================")
common.printNicePolicy(env, bestpolicy, actions2, textsize, "%s: Best Policy\n%s" % (mdpname, envname), "%d-%s-bestpolicy.png" % (envid, mdpid))

# plot goal hole or lost
for i in range(3):
  plt.plot(gammas, np.array(ghls)[:, i], '.-')
plt.legend(['goal', 'hole', 'lost'], loc="best")
plt.ylabel("goals")
plt.xlabel("gamma")
plt.suptitle("Outcome After %d Episodes" % (episodes))
plt.title(envname)
plt.gcf()
plt.savefig("%d-%s-ghl.png" % (envid, mdpid), bbox_inches="tight")
plt.close()

# plot timesteps
plt.plot(gammas, ts, '.-')
plt.xlabel("gamma")
plt.ylabel("mean episode timesteps")
plt.suptitle("Mean Episode Timesteps")
plt.title(envname)
plt.gcf()
plt.savefig("%d-%s-ts.png" % (envid, mdpid), bbox_inches="tight")
plt.close()

# plot convergence
maxlen = 0
for i in ndiffs:
    if len(i) > maxlen:
        maxlen = len(i)
maxlen = maxlen+10
if envid == 2:
  maxlen = int(np.median(list(map(lambda x: len(x), ndiffs))))+10

for i in ndiffs:
    if len(i) < maxlen:
        i.extend([i[-1]]*(maxlen-len(i)))       
    plt.plot(range(maxlen), i[:maxlen])
plt.legend(list(map(lambda x: "gamma = %.1f" % (x), gammas)), loc="best")
plt.ylabel('# of differing policies')
plt.xlabel('iterations')
plt.suptitle('%s: Convergence' % (mdpname))
plt.title(envname)
plt.gcf()
plt.savefig("%d-%s-converge.png" % (envid, mdpid), bbox_inches="tight")
plt.close()