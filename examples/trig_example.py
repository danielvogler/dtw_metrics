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
length_1 = 800
length_2 = 300

### sequence 1
x_1 = np.linspace(0, 8*pi, length_1)
y_1 = np.cos(x_1)
### sequence 2
x_2 = np.linspace(10*pi, 18*pi, length_2)
distortion = np.random.uniform(low=0.8, high=1.0, size=( length_2, )) + 0.2 * np.cos(x_2+pi) - 0.2 * np.cos(x_2*1.25)
y_2 = np.cos(x_2) * distortion * np.linspace(1, 0.5, length_2)


x_11 = np.zeros(length_1)
x_22 = np.zeros(length_2)

xy_1 = np.asarray([x_1,y_1]).T
xy_2 = np.asarray([x_2,y_2]).T

xy_11 = np.asarray([x_11,y_1]).T
xy_22 = np.asarray([x_22,y_2]).T

'''
    compute/plot dtw
'''
### compute dtw
dtw = dtwm.acm( xy_11, xy_22 )
print('DTW: {}'.format(dtw[-1,-1]) )

### compute optimal path
owp = dtwm.optimal_warping_path( dtw )

### plot data and cm
dtwu.plot_sequences( xy_1, xy_2 )
dtwu.plot_cost_matrix( xy_11, xy_22 )
dtwu.plot_acc_cost_matrix( xy_11, xy_22 )

plt.show()
exit()