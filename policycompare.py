import numpy as np
import common
import sys
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def policycompare(envid, policy1, policy2, policyid1, policyid2, annotsize):
  env, envname, _, _, _, actions = common.getEnv(envid)
  envshape = (env.env.nrow, env.env.ncol)
  ne = (policy1 != policy2).reshape(envshape)
  mapnum = np.zeros(envshape)
  mapnum[ne] = 1
  policyactions = common.getPolicyActions(policy1, actions, envshape)
  policyactions[ne] = ''
  title = "%s and %s Policy Comparison\n%s" % (policyid1.upper(), policyid2.upper(), envname)
  figname = "%d-%s%s-policycompare.png" % (envid, policyid1, policyid2)
  common.printHeatMap(mapnum, policyactions, annotsize, title, figname)

pi0 = np.array([0, 3, 0, 3, 0, 0, 2, 0, 3, 1, 0, 0, 0, 2, 1, 0])
vi0 = np.array([0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
q010k = np.array([0, 0, 2, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 2, 0])
q01k = np.array([0, 3, 1, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
vi2 = np.array((3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 0, 1, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 3, 1, 3, 3, 0, 0, 2, 3, 2, 2, 3, 2, 1, 1, 1, 0, 0, 2, 0, 3, 3, 1, 0, 0, 2, 0, 0, 2, 2, 3, 2, 0, 0, 2, 0, 3, 0, 0, 2, 1, 3, 2, 1, 2, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 3, 0, 0, 2, 2, 2, 3, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 1, 0, 3, 1, 3, 2, 1, 1, 1, 2, 1, 0, 0, 1, 3, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 1, 0, 1, 0, 0, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 2, 3, 2, 1, 1, 2, 0, 0, 2, 2, 2, 2, 3, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 2, 2, 0, 0, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 3, 1, 2, 0, 0, 0, 2, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0))
pi2 = np.array((3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 0, 1, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 3, 1, 3, 3, 0, 0, 2, 3, 2, 2, 3, 2, 1, 1, 1, 0, 0, 2, 0, 3, 3, 1, 0, 0, 2, 0, 0, 2, 2, 3, 2, 0, 0, 2, 0, 3, 0, 0, 2, 1, 3, 2, 1, 2, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 3, 0, 0, 2, 2, 2, 3, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 3, 0, 1, 2, 1, 0, 3, 1, 3, 2, 1, 1, 1, 2, 1, 0, 0, 2, 3, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 1, 0, 1, 0, 0, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 2, 3, 2, 1, 1, 2, 0, 0, 2, 2, 2, 2, 3, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 2, 2, 0, 0, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 3, 1, 2, 0, 0, 0, 2, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0))

policycompare(0, pi0, vi0, 'pi', 'vi', 12)
policycompare(0, pi0, q01k, 'pi', 'q1k', 12)
policycompare(0, q01k, vi0, 'q1k', 'vi', 12)
policycompare(0, pi0, q010k, 'pi', 'q10k', 12)
policycompare(0, q010k, vi0, 'q10k', 'vi', 12)
