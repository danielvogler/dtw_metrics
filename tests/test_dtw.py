"""Provide unit test cases."""
import logging
import unittest
from math import pi

import numpy as np
import pytest

from dtwmetrics.dtwmetrics import DTWMetrics

logging.basicConfig(encoding="utf-8", level=logging.INFO)


class TestGeopard(unittest.TestCase):
    """Test example for geopard."""

    def test_trig_function(self):
        """Trigonometric function test."""
        dtwm = DTWMetrics()

        LENGTH_1 = 750
        LENGTH_2 = 400

        x_1 = np.linspace(0, 8 * pi, LENGTH_1)
        y_1 = np.cos(x_1)
        x_2 = np.linspace(10 * pi, 18 * pi, LENGTH_2)
        distortion = (
            np.random.uniform(low=0.8, high=1.0, size=(LENGTH_2,))
            + 0.2 * np.cos(x_2 + pi)
            - 0.2 * np.cos(x_2 * 1.25)
        )
        y_2 = np.cos(x_2) * distortion * np.linspace(1, 0.5, LENGTH_2)

        cm, dtw, owp, warped_query = dtwm.dtwm(
            y_1, y_2, step_pattern="symmetric_p1"
        )
        logging.info("DTW: %f", dtw[-1, -1])

        assert dtw[-1, -1] == pytest.approx(71.5, 0.75)

    def test_phase_shift_function(self):
        """Phase shift function test."""
        dtwm = DTWMetrics()

        # define sequence lengths
        LENGTH_1 = 200
        LENGTH_2 = 250

        # sequence 1
        x_1 = np.linspace(0, 6 * pi, LENGTH_1)
        y_1 = np.cos(x_1)
        # sequence 2
        x_2 = np.linspace(2 * pi, 8 * pi, LENGTH_2)
        y_2 = np.cos(x_2)

        # compute cm, dtw
        cm, dtw, owp, warped_query = dtwm.dtwm(
            y_1, y_2, step_pattern="symmetric_p1"
        )
        logging.info("DTW: %f", dtw[-1, -1])

        assert dtw[-1, -1] == pytest.approx(2.4, 0.1)

    def test_minimal_function(self):
        """Minimal function test."""
        dtwm = DTWMetrics()

        # define sequence lengths
        LENGTH = 100

        # sequence 1
        x = np.linspace(0, 12, LENGTH)
        y_1 = np.cos(x)
        # sequence 2
        y_2 = np.cos(x)
        # sequence 2
        y_3 = np.cos(x) + 0.01

        # sequence 1 vs. 2
        cm, dtw1, owp, warped_query = dtwm.dtwm(
            y_1, y_2, step_pattern="symmetric_p1"
        )
        logging.info("DTW1: %f", dtw1[-1, -1])

        assert dtw1[-1, -1] == pytest.approx(0.0)

        # sequence 1 vs. 3
        cm, dtw2, owp, warped_query = dtwm.dtwm(
            y_1, y_3, step_pattern="symmetric_p1"
        )
        logging.info("DTW2: %f", dtw2[-1, -1])

        assert dtw2[-1, -1] == pytest.approx(0.933, 0.01)
