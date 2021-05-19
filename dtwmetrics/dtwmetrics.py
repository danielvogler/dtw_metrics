'''
Daniel Vogler

References: 
(1) Müller, Meinard. Information retrieval for music and motion. Vol. 2. 
    Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3

'''

import numpy as np


class DTWMetrics:

    ### cost matrix
    def cost_matrix(self, target, estimate):

        ### tile target data
        target_tiled = np.tile(target, (np.size(estimate),1) )
        ### cost matrix
        cost_matrix = np.absolute( ( target_tiled.T - estimate ).T )

        return cost_matrix


    ### accumulated cost matrix
    def acm(self, target, estimate, distance_metric='seuclidean'):

        ### compute cost matrix
        cm = self.cost_matrix(target[0], estimate[0])

        ### sequence lengths
        N, M = cm.shape

        ### initialize
        acm = np.ones( [N, M] )

        ### boundary condition 1
        acm[0,0] = cm[0,0]

        ### From (1) Theorem 4.3
        ### D(n, 1) = \sum_{k=1}^n c(x_k , y_1 ) for n ∈ [1 : N ], 
        ### D(1, m) = \sum_{k=1}^n c(x_1 , y_k ) for m ∈ [1 : M ] and
        ### D(n, m) = min{D(n − 1, m − 1), D(n − 1, m), D(n, m − 1)} + c(x_n , y_m )
        ### for 1 < n ≤ N and 1 < m ≤ M .
        acm[1:,0] = [ acm[n-1,0] + cm[n,0] for n in range(1,N) ]
        acm[0,1:] = [ acm[0,m-1] + cm[0,m] for m in range(1,M) ]
        acm_inner = [ [cm[n,m] + min(acm[n-1,m-1], acm[n-1,m], acm[n,m-1])] for n in range(1,N) for m in range(1,M) ]
        ### fill in acm
        acm[1:,1:] = np.reshape( acm_inner, (N-1, M-1) )

        return acm


    ### optimal warping path owp
    def optimal_warping_path(self, acm):

        p = []

        ### determine acm shape
        N, M = acm.shape
        n = N - 1
        m = M - 1

        p.append([n,m])

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
                optimal_step = min( acm[n-1,m-1], acm[n-1,m], acm[n,m-1] )
                ### progress indices in direction of optimal step
                if optimal_step == acm[n-1, m-1]:
                    n = n - 1
                    m = m - 1
                elif optimal_step == acm[n-1, m]:
                    n = n - 1
                elif optimal_step == acm[n, m-1]:
                    m = m - 1
                else:
                    print('Error in optimal step computation')

            ### append indices of optimal step
            p.append([n,m])
        
        p.append([0,0])

        owp = np.asarray( p )
        owp = np.flip( owp )

        return owp