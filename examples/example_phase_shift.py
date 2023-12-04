"""DTWMetrics example with simple phase shift.

(c) Daniel Vogler
"""
import logging
from math import pi

import numpy as np
from matplotlib import pyplot as plt

from dtwmetrics.dtwmetrics import DTWMetrics
from dtwmetrics.dtwutils import DTWUtils

dtwm = DTWMetrics()
dtwu = DTWUtils()

"""
    trig example
"""
# define sequence lengths
LENGTH_1 = 200
LENGTH_2 = 250

# sequence 1
x_1 = np.linspace(0, 6 * pi, LENGTH_1)
y_1 = np.cos(x_1)
# sequence 2
x_2 = np.linspace(2 * pi, 8 * pi, LENGTH_2)
y_2 = np.cos(x_2)

xy_1 = np.asarray([x_1, y_1]).T
xy_2 = np.asarray([x_2, y_2]).T

"""
    compute/plot dtw
"""
cm, dtw, owp, warped_query = dtwm.dtwm(y_1, y_2, step_pattern="symmetric_p1")
logging.info("DTW: %f", dtw[-1, -1])

# plot data and cm
dtwu.plot_sequences(xy_1, xy_2)
# plot data and cm
dtwu.plot_warped_sequences(y_1, y_2, owp=owp)

# plot data ,cm, acm/dtw and owp
dtwu.plot_matrix(
    xy_1, xy_2, dtw, owp=owp, plot_dim=0, title="Accumulated cost matrix"
)

plt.show()
