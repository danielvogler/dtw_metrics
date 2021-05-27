'''
Daniel Vogler
minimal working example
'''

from dtwmetrics.dtwmetrics import DTWMetrics
from dtwmetrics.dtwutils import DTWUtils
from matplotlib import pyplot as plt

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
### perfect match of cost of 100*0.0 = 0.0
dtw = dtwm.acm( y_1, y_2 )
print('DTW of identical curves: {}'.format(dtw[-1,-1]) )

### cost of 100*0.01 = 1.0
dtw = dtwm.acm( y_1, y_3 )
print('DTW of slightly offset curves: {}'.format(dtw[-1,-1]) )

dtwu.plot_sequences( y_1, y_3 )
plt.show()
