'''
Daniel Vogler

'''

from dtwmetrics.dtwmetrics import DTWMetrics
import numpy as np
from math import pi 
from matplotlib import pyplot as plt

dtwm = DTWMetrics()

'''
    trig example
'''
### define sequence lengths
length_1 = 100
length_2 = 100

### sequence 1
x_1 = np.linspace(0, 4*pi, length_1)
y_1 = np.cos(x_1)
### sequence 2
x_2 = np.linspace(0, 4*pi, length_2)
distortion = np.random.uniform(low=0.8, high=1.0, size=( length_2, )) + np.cos(x_2/4*3) + np.cos(1.5*x_2)*0.2
y_2 = np.cos(x_2) * distortion


'''
    compute/plot dtw
'''
### compute dtw
dtw = dtwm.acm( [y_1,x_1], [y_2,x_2] )
print('DTW: {}'.format(dtw[-1,-1]) )

### compute optimal path
owp = dtwm.optimal_warping_path( dtw )

### plot data and cm
dtwm.plot_sequences( [x_1,y_1], [x_2,y_2] )
dtwm.plot_cost_matrix( [x_1,y_1], [x_2,y_2] )

plt.show()
exit()