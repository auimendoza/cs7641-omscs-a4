import numpy as np
import common
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

for envid in [0, 2]:
  env, envname, _, _, _, _ = common.getEnv(envid)

  mapnum = common.getValMap(env)
  npMAP = np.apply_along_axis(common.byteslist2asciilist, 1, env.env.desc)

  annotsize = 12
  if envid == 0:
    annotsize = 20
  sns.heatmap(mapnum, annot=npMAP, fmt='', annot_kws={"size": annotsize}, cbar=False, cmap="Blues")
  plt.yticks(rotation=0)
  plt.xlabel("")
  plt.ylabel("")
  plt.title("Map of %s" % (envname))
  plt.gcf()
  plt.savefig("%d-map.png" % (envid), bbox_inches='tight')
  plt.close()
