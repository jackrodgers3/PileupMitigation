from scipy.stats import binned_statistic_2d
import random
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

x = [i for i in range(300)]
y = [i for i in np.arange(-1.0, 1.0, (1/150))]

print(len(x))
print(len(y))

values = [random.uniform(-1, 1) for i in range(300)]

binx = [0, 150, 300]
biny = [-1.0, 0.0, 1.0]

ret = binned_statistic_2d(x, y, values, 'std', bins=[5, 5])
print(ret.statistic)
calcs = []
for i in range(5):
    for j in range(5):
        calcs.append(ret.statistic[i][j])

print(ret)

print(calcs)
#b = plt.hist2d(values, ret.statistic, bins = 5, range=[[0, 300], [0, 1]],
#               norm= mpl.colors.LogNorm())
#plt.colorbar(b[3])
#plt.title('Test')
#plt.savefig('test.png')
b = float('nan')
print(b)