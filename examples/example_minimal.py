"""DTWMetrics minimal example.

(c) Daniel Vogler
"""
import logging

import numpy as np
from matplotlib import pyplot as plt

from dtwmetrics.dtwmetrics import DTWMetrics
from dtwmetrics.dtwutils import DTWUtils

dtwm = DTWMetrics()
dtwu = DTWUtils()

"""
    minimal example
"""
# define sequence lengths
LENGTH = 100

# sequence 1
x_1 = x_2 = x_3 = np.linspace(0, 12, LENGTH)
y_1 = np.cos(x_1)
# sequence 2
y_2 = np.cos(x_2)
# sequence 3
y_3 = np.cos(x_2) + 0.01

# compute dtw
# perfect match of cost of 100*0.0 = 0.0
dtw = dtwm.acm(y_1, y_2)
logging.info("DTW of identical curves: %f", dtw[-1, -1])

# cost of 100*0.01 = 1.0
dtw = dtwm.acm(y_1, y_3)
logging.info("DTW of slightly offset curves: %f", dtw[-1, -1])

dtwu.plot_sequences(y_1, y_3)
plt.show()
