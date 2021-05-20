'''
Daniel Vogler

dtw metrics:
- cost matrix
- accumulated cost matrix
- optimal warping path

References: 
(1) Müller, Meinard. Information retrieval for music and motion. Vol. 2. 
    Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3

'''

import numpy as np
from numpy import argmin
from scipy.spatial.distance import cdist

class DTWMetrics:

    ### cost matrix
    def cost_matrix(self, reference, query):

        ### tile reference data
        reference_tiled = np.tile(reference, (np.size(query),1) )
        ### cost matrix
        cost_matrix = np.absolute( ( reference_tiled.T - query ).T )

        return cost_matrix


    ### accumulated cost matrix
    def acm(self, reference, query, distance_metric='cityblock'):

        ### compute cost matrix
        # cm = self.cost_matrix(reference[0], query[0])
        cm = cdist(reference, query, metric=distance_metric)

        ### sequence lengths
        N, M = cm.shape

        ### initialize
        acm = np.zeros( [N, M] )

        ### boundary condition 1
        acm[0,0] = cm[0,0]

        ### From (1) Theorem 4.3
        ### D(n, 1) = \sum_{k=1}^n c(x_k , y_1 ) for n ∈ [1 : N ], 
        ### D(1, m) = \sum_{k=1}^n c(x_1 , y_k ) for m ∈ [1 : M ] and
        acm[1:,0] = [ acm[n-1,0] + cm[n,0] for n in range(1,N) ]
        acm[0,1:] = [ acm[0,m-1] + cm[0,m] for m in range(1,M) ]

        ### for 1 < n ≤ N and 1 < m ≤ M .
        ### D(n, m) = min{D(n − 1, m − 1), D(n − 1, m), D(n, m − 1)} + c(x_n , y_m )
        for n in range(1, N):
            for m in range(1, M):
                acm[n, m] = cm[n, m] + min( acm[n-1, m], acm[n, m-1], acm[n-1, m-1]) 

        return acm


    ### optimal warping path owp
    def optimal_warping_path(self, acm):

        p = []

        ### determine acm shape
        N, M = acm.shape
        n = N - 1
        m = M - 1

        p.append([N,M])

        ### compute in reverse order
        ### From (1) Algorithm: OptimalWarpingPath
        while n > 0 and m > 0:
            ### check if acm bounds are reached
            if n == 1:
                m = m - 1
            elif m == 1:
                n = n - 1
            else:
                ### compute direction of optimal step
                optimal_step = []
                optimal_step = argmin( [ acm[n-1,m-1], acm[n-1,m], acm[n,m-1] ] )
                ### progress indices in direction of optimal step
                if optimal_step == 0:
                    n = n - 1
                    m = m - 1
                elif optimal_step == 1:
                    n = n - 1
                elif optimal_step == 2:
                    m = m - 1
                else:
                    print('Error in optimal step computation')

            ### append indices of optimal step
            p.append([n,m])
        
        p.append([0,0])

        owp = np.asarray( p )
        owp = np.flip( owp )

        return owp