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

xy_1 = np.asarray([x_1,y_1]).T
xy_2 = np.asarray([x_2,y_2]).T
xy_3 = np.asarray([x_3,y_3]).T

### compute dtw
dtw = dtwm.acm( xy_1, xy_2 )
print('DTW of identical curves: {}'.format(dtw[-1,-1]) )

dtw = dtwm.acm( xy_1, xy_3 )
print('DTW of slightly offset curves: {}'.format(dtw[-1,-1]) )