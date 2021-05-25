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
length_1 = 750
length_2 = 400

### sequence 1
x_1 = np.linspace(0, 8*pi, length_1)
y_1 = np.cos(x_1)
### sequence 2
x_2 = np.linspace(10*pi, 18*pi, length_2)
distortion = np.random.uniform(low=0.8, high=1.0, size=( length_2, )) + 0.2 * np.cos(x_2+pi) - 0.2 * np.cos(x_2*1.25)
y_2 = np.cos(x_2) * distortion * np.linspace(1, 0.5, length_2)

xy_1 = np.asarray([y_1]).T
xy_2 = np.asarray([y_2]).T

'''
    compute/plot dtw
'''
### compute cm, dtw
dtw = dtwm.acm( y_1, y_2, step_pattern='symmetric_p1' )
cm = dtwm.cm( y_1, y_2 )


print('DTW: {}'.format(dtw[-1,-1]) )

### compute optimal path
owp = dtwm.optimal_warping_path( dtw )

### plot data and cm
dtwu.plot_sequences( xy_1, xy_2 )
dtwu.plot_matrix( xy_1, xy_2, dtw, owp=owp, plot_dim=0, title="Accumulated cost matrix" )
dtwu.plot_matrix( xy_1, xy_2, cm, owp=owp, plot_dim=0, title="Cost matrix" )
dtwu.plot_warped_sequences( xy_1, xy_2, owp )

plt.show()
exit()