import gym
import numpy as np
from mdptoolbox.mdp import ValueIteration
from mdplib import PolicyIteration, QLearning
import common
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def Usage():
  print("Usage:")
  print("python %s <envid> <exploreinterval>" % (sys.argv[0]))
  sys.exit(1)

if len(sys.argv) != 3:
  Usage()

mdpid = "q"
mdpname = "Q Learning"
envid = int(sys.argv[1])
exploreinterval = int(sys.argv[2])

maxiter = 2000000
interval = 250000
gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
episodes = 1000
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

env, envname, P, R, actions, actions2 = common.getEnv(envid)

print("* %s *" % (envname))
print(mdpname)

aiters = []
aghls = []
ats = []
amd = []
bestgoal = 0
bestpolicy = None
bestpolicyparams = {}

print("Running ...")
for alpha in alphas:
  iters = []
  ghls = []
  ts = []
  md = []
  for gamma in gammas:
    #print("gamma: %.1f, alpha: %s" % (gamma, str(alpha)))
    sys.stdout.write('.')
    func = QLearning(P, R, gamma, maxiter, interval, alpha, exploreinterval)
    func.run()
    ighl = []
    its = []
    for i, policy in enumerate(func.policies):
      timesteps, gtimesteps, ghl = common.runPolicy(env, episodes, policy)
      if ghl[0] > bestgoal:
        bestgoal = ghl[0]
        bestpolicy = policy
        bestpolicyparams['gamma'] = gamma
        bestpolicyparams['alpha'] = alpha
        bestpolicyparams['iterations'] = func.iterations[i]
        bestpolicyparams['elapsedtime'] = func.elapsedtimes[i]
        bestpolicyparams['meangtimesteps'] = np.mean(gtimesteps)
      ighl.append(ghl)
      its.append(np.mean(timesteps))
    iters.append(func.iterations)
    ghls.append(ighl)
    ts.append(its)
    md.append(func.mean_discrepancy)
  aiters.append(iters)
  aghls.append(ghls)
  ats.append(ts)
  amd.append(md)

# plot best policy 
textsize = 12
if envid == 0:
  textsize = 20
print("== best policy ==")
print("explore interval = %d" % (exploreinterval))
print("goals = %d" % (bestgoal))
print("gamma = %.1f" % (bestpolicyparams['gamma']))
print("alpha = %.1f" % (bestpolicyparams['alpha']))
print("iterations = %d" % (bestpolicyparams['iterations']))
print("elapsed time = %.3f" % (bestpolicyparams['elapsedtime']))
print("mean timesteps to goal = %.3f" % (bestpolicyparams['meangtimesteps']))
print("=================")
common.printNicePolicy(env, bestpolicy, actions2, textsize, "%s: Best Policy\n%s" % (mdpname, envname), "%d-%s-bestpolicy.png" % (envid, mdpid))
print(bestpolicy)

# plot iterations, params vs goal/discrepancy
iterations = aiters[0][0]
ghls = []
aghl = np.array(aghls)
for i in range(aghl.shape[0]):
    for j in range(aghl.shape[1]):
        for k in range(aghl.shape[2]):
            ghls.append([alphas[i], gammas[j], iterations[k], aghl[i,j,k,0], aghl[i,j,k,1], aghl[i,j,k,2], np.array(amd)[i,j,k], np.array(ats)[i,j,k]])
ghlpd = pd.DataFrame(ghls, columns=['alpha', 'gamma', 'i', 'goal', 'hole', 'lost', 'md', 'ts'])
ghlpd['param'] = ghlpd.apply(lambda row: 'alpha=' + str(row['alpha']) + ', gamma=' + str(row['gamma']), axis=1)

g = sns.FacetGrid(ghlpd, col="alpha", hue="gamma", col_wrap=3, legend_out=False)
g = g.map(plt.plot, "i", "goal", marker=".")
g.add_legend()
g.set_xlabels('iterations')
g.set_ylabels('goals')
g.fig.subplots_adjust(top=0.7)
g.fig.suptitle('Goals After %d Episodes\n%s' % (episodes, envname))
g.set_xticklabels(rotation=90)
plt.gcf()
plt.savefig("%d-%s-%d-goal-it.png" % (envid, mdpid, exploreinterval), bbox_inches="tight")
plt.close()

g = sns.FacetGrid(ghlpd, col="alpha", hue="gamma", col_wrap=3, legend_out=False)
g = g.map(plt.plot, "i", "md", marker=".")
g.add_legend()
g.set_xlabels('iterations')
g.set_ylabels('Q mean discrepancy')
g.fig.subplots_adjust(top=0.7)
g.fig.suptitle('Q Mean Discrepancy\n%s' % (envname))
g.set_xticklabels(rotation=90)
plt.gcf()
plt.savefig("%d-%s-%d-md-it.png" % (envid, mdpid, exploreinterval), bbox_inches="tight")
plt.close()

g = sns.FacetGrid(ghlpd, col="alpha", hue="gamma", col_wrap=3, legend_out=False)
g = g.map(plt.plot, "i", "ts", marker=".")
g.add_legend()
g.set_xlabels('iterations')
g.set_ylabels('timesteps')
g.fig.subplots_adjust(top=0.7)
g.fig.suptitle('Timesteps to Goal or Hole\n%s' % (envname))
g.set_xticklabels(rotation=90)
g.set(ylim=(0,200))
plt.gcf()
plt.savefig("%d-%s-%d-ts-it.png" % (envid, mdpid, exploreinterval), bbox_inches="tight")
plt.close()