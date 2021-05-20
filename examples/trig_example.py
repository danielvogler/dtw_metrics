'''
Daniel Vogler

'''

from dtwmetrics.dtwmetrics import DTWMetrics
from dtwmetrics.dtwutils import DTWUtils

import numpy as np
from math import pi 
from matplotlib import pyplot as plt

dtwm = DTWMetrics()
dtwu = DTWUtils()

'''
    trig example
'''
### define sequence lengths
length_1 = 100
length_2 = 500

### sequence 1
x_1 = np.linspace(0+pi/4, 4*pi+pi/4, length_1)
y_1 = np.cos(x_1)
### sequence 2
x_2 = np.linspace(0, 4*pi, length_2)
distortion = np.random.uniform(low=0.9, high=1.1, size=( length_2, )) + 0.1 * np.cos(x_2/1.25)# + 0.1 * np.cos(1.25*x_2)
y_2 = np.cos(x_2) * distortion

xy_1 = np.asarray([x_1,y_1]).T
xy_2 = np.asarray([x_2,y_2]).T

'''
    compute/plot dtw
'''
### compute dtw
dtw = dtwm.acm( xy_1, xy_2 )
print('DTW: {}'.format(dtw[-1,-1]) )

### compute optimal path
owp = dtwm.optimal_warping_path( dtw )

### plot data and cm
dtwu.plot_sequences( xy_1, xy_2 )
dtwu.plot_cost_matrix( xy_1, xy_2 )
dtwu.plot_acc_cost_matrix( xy_1, xy_2 )

plt.show()
exit()