"""DTWMetrics example for trigonometrics functions.

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
LENGTH_1 = 750
LENGTH_2 = 400

# sequence 1
x_1 = np.linspace(0, 8 * pi, LENGTH_1)
y_1 = np.cos(x_1)
# sequence 2
x_2 = np.linspace(10 * pi, 18 * pi, LENGTH_2)
distortion = (
    np.random.uniform(low=0.8, high=1.0, size=(LENGTH_2,))
    + 0.2 * np.cos(x_2 + pi)
    - 0.2 * np.cos(x_2 * 1.25)
)
y_2 = np.cos(x_2) * distortion * np.linspace(1, 0.5, LENGTH_2)

xy_1 = np.asarray([y_1]).T
xy_2 = np.asarray([y_2]).T

"""
    compute/plot dtw
"""
# compute cm, dtw
cm, dtw, owp, warped_query = dtwm.dtwm(y_1, y_2, step_pattern="symmetric_p1")
logging.info("DTW: %f", dtw[-1, -1])

# plot data ,cm, acm/dtw and owp
dtwu.plot_matrix(
    xy_1, xy_2, dtw, owp=owp, plot_dim=0, title="Accumulated cost matrix"
)
dtwu.plot_matrix(xy_1, xy_2, cm, owp=owp, plot_dim=0, title="Cost matrix")
dtwu.plot_warped_sequences(xy_1, xy_2, owp)

plt.show()
