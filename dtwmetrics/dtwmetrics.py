"""Dynamic time warping metrics.

(c) Daniel Vogler

dtw metrics:
- cost matrix
- accumulated cost matrix
- optimal warping path

References:
(1) Müller, Meinard. Information retrieval for music and motion. Vol. 2.
    Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3

"""
import logging
from typing import Tuple

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist


class DTWMetrics:
    """Dynamic time warping metrics."""

    def __init__(self):
        """Init."""
        return

    def dtwm(
        self,
        reference: np.ndarray,
        query: np.ndarray,
        distance_metric: str = "euclidean",
        step_pattern: str = "symmetric_p0",
        sequence: str = "whole",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute dynamic time warping metrics.

        Args:
            reference (np.ndarray): _description_
            query (np.ndarray): _description_
            distance_metric (str, optional): Distance metric between data
                points. Defaults to "euclidean".
            step_pattern (str, optional): Step pattern of walking path.
                Defaults to "symmetric_p0".
            sequence (str, optional): _description_. Defaults to "whole".

        Returns:
            Tuple: Dynamic time warping metrics such as cost matrix
        """
        logging.info("Compute dynamic time warping metrics")

        cm = self.cm(X=reference, Y=query, distance_metric=distance_metric)

        acm = self.acm(
            reference=reference,
            query=query,
            distance_metric=distance_metric,
            step_pattern=step_pattern,
            sequence=sequence,
        )

        # match whole sequence or only sub-sequence
        if sequence == "sub":
            b, delta_b = self.compute_similar_subsequences(acm)
            owp = self.optimal_warping_path(acm)
            warped_query = self.warped_sequence(query, owp)

        else:
            owp = self.optimal_warping_path(acm)
            warped_query = self.warped_sequence(query, owp)

        return cm, acm, owp, warped_query

    def compute_similar_subsequences(
        self, acm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute similar subsequences.

        distance function Δ:[1:M] → R, Δ(b) := D(N, b),
        assigns each index b ∈ [1:M] the minimal DTW distance Δ(b) that
        can be achieved between X and a subsequence Y (a:b)
        of Y ending in y_b .

        Args:
            acm (np.ndarray): accumulated cost matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: descriptors for subsequences
        """
        logging.info("Compute similar subsequences.")

        delta_b = acm[-1, :]

        logging.debug("Searching local minima")
        local_min = argrelextrema(delta_b, np.less)[0]

        b = None

        # reset all local minima
        for b in local_min:
            owp = self.optimal_warping_path(acm, b=b)
            logging.info(
                "Optimal walking path for similar subsequences %f", owp
            )

        return b, delta_b

    # cost matrix calculation
    def cm(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        distance_metric: str = "euclidean",
        method: str = "cdist",
    ) -> np.ndarray:
        """Compute cost matrix by comparing 2 sequences.

        Args:
            X (np.ndarray): Sequence 1
            Y (np.ndarray): Sequence 2
            distance_metric (str, optional): Distance metric between
                points. Defaults to "euclidean".
            method (str, optional): Method to use for distance calc.
                Defaults to "cdist".

        Returns:
            np.ndarray: cost matrix
        """
        logging.info("Computing cost matrix with %s", distance_metric)

        X = self.dim_check(X)
        Y = self.dim_check(Y)

        if method == "cdist":
            logging.info("Using Scipy's 'cdist' method to compute distance")
            cm = cdist(X, Y, metric=distance_metric)

        else:
            logging.info("Use own method to compute distance")

            X = np.asarray(X, order="c")
            Y = np.asarray(Y, order="c")

            # dimensions
            X_sh = X.shape
            Y_sh = Y.shape
            logging.debug("Cost matrix dimensions (%f/%f)", X_sh, Y_sh)

            logging.debug("Initialize cost matrix")
            cm = np.empty((X_sh[0], Y_sh[0]), dtype=np.double)

            # function string
            dm_str = str("distance_" + distance_metric)
            dm_func = getattr(self, dm_str)

            # create cost matrix
            for i in range(0, X_sh[0]):
                for j in range(0, Y_sh[0]):
                    cm[i, j] = dm_func(X[i], Y[j])

        return cm

    def dim_check(self, x: np.ndarray) -> np.ndarray:
        """Check dimensionality.

        check if 1D and convert to vector array if so

        Args:
            x (np.ndarray): input sequence

        Returns:
            np.ndarray: output sequence
        """
        logging.info("Check dimensionality")

        # check dimensionality
        x = np.atleast_2d(x)

        # transform for cdist if 1D
        if x.shape[0] == 1:
            x = x.T

        return x

    def distance_cityblock(self, a: np.ndarray, b: np.ndarray):
        """Compute manhattan or cityblock distance."""
        dist = np.sum(np.absolute(a - b))
        return dist

    def distance_euclidean(self, a: np.ndarray, b: np.ndarray):
        """Compute euclidean distance."""
        dist = np.sqrt(np.sum((a - b) ** 2))
        return dist

    def acm(
        self,
        reference: np.ndarray,
        query: np.ndarray,
        distance_metric="euclidean",
        step_pattern="symmetric_p0",
        sequence="whole",
    ) -> np.ndarray:
        """Generate accumulated cost matrix.

        Args:
            reference (np.ndarray): sequence 1
            query (np.ndarray): sequence 2
            distance_metric (str, optional): distance metric.
                Defaults to "euclidean".
            step_pattern (str, optional):step pattern.
                Defaults to "symmetric_p0".
            sequence (str, optional): whole or part of sequence.
                Defaults to "whole".

        Returns:
            np.ndarray: accumulated cost matrix
        """
        logging.info(
            "Computing accumulated cost matrix with %s", distance_metric
        )

        cm = self.cm(reference, query, distance_metric)

        # function string
        step_pattern_str = str("step_" + step_pattern)
        step_pattern_func = getattr(self, step_pattern_str)

        # execute step path
        acm = step_pattern_func(cm, sequence=sequence)

        return acm

    def step_symmetric_p0(
        self, cm: np.ndarray, sequence="whole"
    ) -> np.ndarray:
        """Compute accumulated cost matrix for symmetric p0 pattern.

        Args:
            cm (np.ndarray): cost matrix
            sequence (str, optional): sequence part.
                Defaults to "whole".

        Raises:
            Exception: If sequence type is undefined

        Returns:
            np.ndarray: accumulated cost matrix
        """
        logging.info(
            "Compute accumulated cost matrix for symmetric p0 pattern"
        )

        # sequence lengths
        N, M = cm.shape

        # initialize
        acm = np.zeros([N, M])

        # boundary condition 1
        acm[0, 0] = cm[0, 0]

        # From (1) Theorem 4.3
        # D(n, 1) = \sum_{k=1}^n c(x_k , y_1 ) for n ∈ [1 : N ],
        for n in range(1, N):
            acm[n, 0] = acm[n - 1, 0] + cm[n, 0]

        # compute acm for whole or sub-sequence
        if sequence == "whole":
            # D(1, m) = \sum_{k=1}^n c(x_1 , y_k ) for m ∈ [1 : M ] and
            for m in range(1, M):
                acm[0, m] = acm[0, m - 1] + cm[0, m]

        elif sequence == "sub":
            # D(1, m) = c(x_1 , y_m ) for m ∈ [1 : M ] and
            for m in range(1, M):
                acm[0, m] = cm[0, m]

        else:
            raise ValueError("Undefined sequence type")

        # for 1 < n ≤ N and 1 < m ≤ M .
        # D(n, m) = min{D(n − 1, m − 1), D(n − 1, m),
        #   D(n, m − 1)} + c(x_n , y_m )
        for n in range(1, N):
            for m in range(1, M):
                acm[n, m] = cm[n, m] + min(
                    acm[n - 1, m], acm[n, m - 1], acm[n - 1, m - 1]
                )

        return acm

    def step_symmetric_p1(
        self, cm: np.ndarray, sequence: str = "whole"
    ) -> np.ndarray:
        """Compute accumulated cost matrix for symmetric p1 pattern.

        Args:
            cm (np.ndarray): cost matrix

        Raises:
            Exception: If sequence type is undefined

        Returns:
            np.ndarray: accumulated cost matrix
        """
        logging.info(
            "Compute accumulated cost matrix for symmetric p1 pattern"
        )
        logging.info("Sequence type %s", sequence)

        # sequence lengths
        N, M = cm.shape

        # check if sequences differ at most by factor of 2
        if N > 2 * M:
            raise ValueError("Reference length to query length ratio > 2")

        if M > 2 * N:
            raise ValueError("Query length to reference length ratio > 2")

        # initialize
        acm = np.zeros([N, M])

        #
        # D(n, m) = min{D(n − 1, m − 1), D(n − 2, m − 1),
        #   D(n − 1, m − 2)} + c(x n , y m )
        #
        # with initial values:
        # D(0, 0) := 0,
        acm[0, 0] = 0

        # D(1, 1) := c(x_1 , y_1 ),
        acm[1, 1] = cm[1, 1]

        # D(n, 0) := ∞ for n ∈ [1 : N],
        for n in range(1, N):
            acm[n, 0] = np.inf

        # D(n, 1) := ∞ for n ∈ [2 : N],
        for n in range(2, N):
            acm[n, 1] = np.inf

        # D(0, m) := ∞ for m ∈ [1 : M ], and
        for m in range(1, M):
            acm[0, m] = np.inf

        # D(1, m) := ∞ for m ∈ [2 : M ].
        for m in range(2, M):
            acm[1, m] = np.inf

        # D(n, m) = min{D(n − 1, m − 1), D(n − 2, m − 1),
        #   D(n − 1, m − 2)} + c(x n , y m )
        for n in range(2, N):
            for m in range(2, M):
                acm[n, m] = (
                    min(
                        acm[n - 1, m - 1], acm[n - 2, m - 1], acm[n - 1, m - 2]
                    )
                    + cm[n, m]
                )

        return acm

    def optimal_warping_path(self, acm: np.ndarray, b=None) -> np.ndarray:
        """Compute optimal warping path.

        Args:
            acm (np.ndarray): accumulated cost matrix
            b (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            np.ndarray: optimal warping path
        """
        logging.info("Compute optimal warping path (owp)")

        # determine acm shape
        N, M = acm.shape

        # if subsequence owp requested
        if b:
            if b < M:
                M = b
            else:
                raise ValueError(
                    "Subsequence length must be below total array length"
                )
        else:
            logging.info("Matching entire query sequence")

        # move one entry in reverse
        n = N - 1
        m = M - 1

        # owp to be populated in reverse
        p = []
        p.append([N, M])

        # compute in reverse order
        # From (1) Algorithm: OptimalWarpingPath
        while n > 0 and m > 0:
            # check if acm bounds are reached
            if n == 1:
                m = m - 1
            elif m == 1:
                n = n - 1
            else:
                # compute direction of optimal step
                optimal_step = []
                optimal_step = np.argmin(
                    [acm[n - 1, m - 1], acm[n - 1, m], acm[n, m - 1]]
                )
                # progress indices in direction of optimal step
                if optimal_step == 0:
                    n = n - 1
                    m = m - 1
                elif optimal_step == 1:
                    n = n - 1
                elif optimal_step == 2:
                    m = m - 1
                else:
                    raise ValueError("Error in optimal step computation")

            # append indices of optimal step
            p.append([n, m])

        # B.C.
        p.append([0, 0])

        owp = np.asarray(p)
        owp = np.flip(owp)

        return owp

    # warped sequence
    def warped_sequence(
        self, sequence: np.ndarray, owp: np.ndarray
    ) -> np.ndarray:
        """Compute warped sequence.

        Args:
            sequence (np.ndarray): sequence
            owp (np.ndarray): optimal warping path

        Returns:
            np.ndarray: warped sequence (to achieve match with sequence 2)
        """
        logging.info("Compute warped sequence")
        warped_sequence = [
            [owp[i, 1], sequence[owp[i, 0]]] for i in range(owp.shape[0] - 1)
        ]
        warped_sequence = np.asarray(warped_sequence, dtype="object")

        return warped_sequence
