"""DTWUtils like plotting etc.

(c) Daniel Vogler

Utilities:
- plotting
"""
import logging

import numpy as np
from matplotlib import pyplot as plt

from dtwmetrics.dtwmetrics import DTWMetrics

dtwm = DTWMetrics()


class DTWUtils:
    """Util class for dynamic time warping."""

    def plot_sequences(self, reference: np.ndarray, query: np.ndarray) -> None:
        """Plot two sequences.

        Args:
            reference (np.ndarray): sequence 1
            query (np.ndarray): sequence 2
        """
        logging.info("Plot two sequences.")

        reference = dtwm.dim_check(reference)
        query = dtwm.dim_check(query)

        plt.figure(
            num=None, figsize=(16, 8), dpi=80, facecolor="w", edgecolor="k"
        )
        font = {"size": 14}
        plt.rc("font", **font)

        # reference dim check
        if min(reference.shape) == 1:
            plt.plot(reference, marker=".", c="k", label="Reference")
        else:
            plt.scatter(
                reference[:, 0],
                reference[:, 1],
                s=500,
                marker=".",
                c="k",
                label="Reference",
            )

        # query dim check
        if min(query.shape) == 1:
            plt.plot(query, marker=".", c="r", label="Query")
        else:
            plt.scatter(
                query[:, 0],
                query[:, 1],
                s=500,
                marker=".",
                c="r",
                label="Query",
            )
        plt.legend(loc="upper center")
        plt.xlabel("Time [-]")
        plt.ylabel("Value [-]")
        plt.title("Time sequence")

    def plot_warped_sequences(
        self, reference: np.ndarray, query: np.ndarray, owp: np.ndarray
    ) -> None:
        """Plot warped sequences.

        Args:
            reference (np.ndarray): sequence 1
            query (np.ndarray): sequence 2
            owp (np.ndarray): optimal warping path to overlay
                sequence 1 on sequence 2
        """
        reference = dtwm.dim_check(reference)
        query = dtwm.dim_check(query)

        plt.figure(
            num=None, figsize=(16, 8), dpi=80, facecolor="w", edgecolor="k"
        )
        font = {"size": 14}
        plt.rc("font", **font)

        # reference dim check
        if min(reference.shape) == 1:
            plt.plot(
                reference,
                marker=".",
                c="k",
                label="Reference",
                linestyle="None",
            )
        else:
            plt.scatter(
                reference[:, 0],
                reference[:, 1],
                marker=".",
                c="k",
                label="Reference",
            )

        # query dim check
        if min(query.shape) == 1:
            plt.plot(query, marker=".", c="r", label="Query", linestyle="None")
        else:
            plt.scatter(
                query[:, 0], query[:, 1], marker=".", c="r", label="Query"
            )

        # warped sequence
        warped_query = dtwm.warped_sequence(query, owp)
        if min(warped_query.shape) == 1:
            plt.plot(
                warped_query,
                marker=".",
                c="b",
                label="Warped query",
                linestyle="None",
            )
        else:
            plt.scatter(
                warped_query[:, 0],
                warped_query[:, 1],
                marker=".",
                c="b",
                label="Warped query",
            )

        plt.legend(loc="upper center")
        plt.xlabel("Index [-]")
        plt.ylabel("Value [-]")
        plt.title("Sequences")

    def plot_matrix(
        self,
        reference: np.ndarray,
        query: np.ndarray,
        matrix: str,
        owp=None,
        plot_dim: int = 1,
        title: str = "Matrix",
    ) -> None:
        """Plot different matrices.

        Args:
            reference (np.ndarray): sequence 1
            query (np.ndarray): sequence 2
            matrix (str): matrix
            owp (_type_, optional): optimal warping path. Defaults to None.
            distance_metric (str, optional): distance metric.
                Defaults to "euclidean".
            plot_dim (int, optional): dimension. Defaults to 1.
            title (str, optional): plot title. Defaults to "Matrix".
        """
        reference = dtwm.dim_check(reference)
        query = dtwm.dim_check(query)

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        font = {"size": 14}
        plt.rc("font", **font)
        grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
        x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax)

        main_ax.set_title(title)
        # plot passed matrix

        # plot owp if given
        if "owp" in locals():
            try:
                main_ax.plot(owp[:, 0], owp[:, 1], color="w")
            except ValueError:
                logging.debug("OPW plotting not possible")

        main_ax.pcolormesh(matrix)
        main_ax.yaxis.tick_right()
        main_ax.xaxis.tick_top()

        # plots on the attached axes
        x_plot.plot(
            np.linspace(0, len(query[:, plot_dim]), len(query[:, plot_dim])),
            query[:, plot_dim],
            color="gray",
        )
        x_plot.invert_yaxis()
        x_plot.set_ylim([-1.5, 1.5])
        x_plot.set_xlabel("Query [-]")
        # y-axis
        y_plot.plot(
            reference[:, plot_dim],
            np.linspace(
                0, len(reference[:, plot_dim]), len(reference[:, plot_dim])
            ),
            color="gray",
        )
        y_plot.invert_xaxis()
        y_plot.set_xlim([1.5, -1.5])
        y_plot.set_ylabel("Reference [-]")

    def plot_delta_b(self, acm: np.ndarray) -> None:
        """Plot different deltas for subqueries.

        Args:
            acm (np.ndarray): accumulated cost matrix
        """
        b, delta_b = dtwm.compute_similar_subsequences(acm)
        plt.figure(
            num=None, figsize=(16, 8), dpi=80, facecolor="w", edgecolor="k"
        )
        font = {"size": 14}
        plt.rc("font", **font)

        plt.plot(delta_b, marker=".", c="r", label="delta_b")
        plt.legend(loc="upper center")
        plt.xlabel("Time [-]")
        plt.ylabel("Value [-]")
        plt.title("delta b")
