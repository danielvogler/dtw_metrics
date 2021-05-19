'''
Daniel Vogler
minimal working example
'''

from dtwmetrics.dtwmetrics import DTWMetrics
from dtwmetrics.dtwutils import DTWUtils

import numpy as np

dtwm = DTWMetrics()
dtwu = DTWUtils()

'''
    minimal example
'''
### define sequence lengths
length = 100

### sequence 1
x_1 = x_2 = x_3 = np.linspace(0, 12, length)
y_1 = np.cos(x_1)
### sequence 2
y_2 = np.cos(x_2)
### sequence 3
y_3 = np.cos(x_2) + 0.01

### compute dtw
dtw = dtwm.acm( [y_1,x_1], [y_2,x_2] )
print('DTW of identical curves: {}'.format(dtw[-1,-1]) )

dtw = dtwm.acm( [y_1,x_1], [y_3,x_3] )
print('DTW of slightly offset curves: {}'.format(dtw[-1,-1]) )