import numpy as np
from matplotlib import pyplot as plt
import pandas

l = []
with open("log.txt","r") as f:
    for line in f:
        l.append(float(line.strip()))

plt.plot(pandas.rolling_mean(np.array(l),100,1))
plt.show()
