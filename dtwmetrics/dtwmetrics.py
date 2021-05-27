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
from scipy.spatial.distance import cdist

class DTWMetrics:


    ### compute all dtwm_metrics
    def dtwm(self, reference, query, distance_metric='euclidean', step_pattern='symmetric_p0', sequence='whole' ):

        ### compute cost matrix cm, 
        ### accumulated cost matrix acm,
        ### optimal warping path owp and
        ### warped sequence
        cm = self.cm( reference, query, distance_metric=distance_metric )
        acm = self.acm( reference, query, distance_metric=distance_metric, step_pattern=step_pattern )
        owp = self.optimal_warping_path( acm )   
        warped_query = self.warped_sequence(query, owp)      

        return cm, acm, owp, warped_query


    ### cost matrix calculation
    def cm(self, X, Y, distance_metric='euclidean', method='cdist'):

        print('\tComputing cost matrix ({})\n'.format(distance_metric) )

        X = self.dim_check( X )
        Y = self.dim_check( Y )

        ### use scipy cdist
        if method == 'cdist':

            cm = cdist(X, Y, metric=distance_metric)

        ### use own cost matrix calculation
        else:

            X = np.asarray(X, order='c')
            Y = np.asarray(Y, order='c')

            ### dimensions
            X_sh = X.shape
            Y_sh = Y.shape
        
            ### initialize
            cm = np.empty( (X_sh[0], Y_sh[0]), dtype=np.double )

            ### function string
            dm_str = str('distance_' + distance_metric)
            dm_func = getattr(self, dm_str)

            ### create cost matrix
            for i in range(0, X_sh[0] ):
                for j in range(0, Y_sh[0] ):
                    cm[i, j] = dm_func(X[i], Y[j]) 

        return cm

            
    ### check if 1D and convert to vector array if so
    def dim_check(self, x):

        ### check dimensionality
        x = np.atleast_2d( x )

        ### transform for cdist if 1D
        if x.shape[0] == 1:
            x = x.T

        return x


    ### manhattan or cityblock distance
    def distance_cityblock(self, a, b):

        dist = np.sum( np.absolute( a - b ) )

        return dist


    ### euclidean distance
    def distance_euclidean(self, a, b):

        dist = np.sqrt( np.sum( (a - b)**2 ) )
        
        return dist


    ### accumulated cost matrix
    def acm(self, reference, query, distance_metric='euclidean', step_pattern='symmetric_p0' ):

        print('\tComputing accumulated cost matrix ({})\n'.format(distance_metric) )

        ### call cm
        cm = self.cm(reference, query, distance_metric)

        ### function string
        step_pattern_str = str('step_' + step_pattern)
        step_pattern_func = getattr(self, step_pattern_str)

        ### execute step path
        acm = step_pattern_func(cm)

        return acm


    ### step pattern: symmetric p0
    def step_symmetric_p0(self, cm, sequence='whole' ):

        ### sequence lengths
        N, M = cm.shape

        ### initialize
        acm = np.zeros( [N, M] )

        ### boundary condition 1
        acm[0,0] = cm[0,0]

        ### From (1) Theorem 4.3
        ### D(n, 1) = \sum_{k=1}^n c(x_k , y_1 ) for n ∈ [1 : N ], 
        for n in range(1,N):
            acm[n,0] = acm[n-1,0] + cm[n,0]
            
        ### compute acm for whole or sub-sequence
        if sequence == 'whole':

            ### D(1, m) = \sum_{k=1}^n c(x_1 , y_k ) for m ∈ [1 : M ] and
            for m in range(1,M):
                acm[0,m] = acm[0,m-1] + cm[0,m]

        elif sequence == 'sub':

            ### D(1, m) = c(x_1 , y_m ) for m ∈ [1 : M ] and
            for m in range(1,M):
                acm[0,m] = cm[0,m]

        else:

            print('\tERROR: Undefined sequence type\n')

        ### for 1 < n ≤ N and 1 < m ≤ M .
        ### D(n, m) = min{D(n − 1, m − 1), D(n − 1, m), D(n, m − 1)} + c(x_n , y_m )
        for n in range(1, N):
            for m in range(1, M):
                acm[n, m] = cm[n, m] + min( acm[n-1, m], acm[n, m-1], acm[n-1, m-1]) 

        return acm


    ### step pattern: symmetric p1
    def step_symmetric_p1(self, cm):

        print('\t--> step pattern: symmetric P1\n')

        ### sequence lengths
        N, M = cm.shape

        ### check if sequences differ at most by factor of 2
        if N > 2*M:
            print('Reference length differs from query length by more than a factor of 2!')

        elif M > 2*N:
            print('Query length differs from reference length by more than a factor of 2!')

        ### initialize
        acm = np.zeros( [N, M] )

        ###
        ### D(n, m) = min{D(n − 1, m − 1), D(n − 2, m − 1), D(n − 1, m − 2)} + c(x n , y m )
        ###
        ### with initial values:
        ### D(0, 0) := 0,
        acm[0, 0] = 0

        ### D(1, 1) := c(x_1 , y_1 ), 
        acm[1,1] = cm[1,1]

        ### D(n, 0) := ∞ for n ∈ [1 : N], 
        for n in range(1,N):
            acm[n,0] = np.inf

        ### D(n, 1) := ∞ for n ∈ [2 : N],
        for n in range(2,N):
            acm[n,1] = np.inf

        ### D(0, m) := ∞ for m ∈ [1 : M ], and 
        for m in range(1,M):
            acm[0,m] = np.inf
            
        ### D(1, m) := ∞ for m ∈ [2 : M ].
        for m in range(2,M):
            acm[1,m] = np.inf

        ### D(n, m) = min{D(n − 1, m − 1), D(n − 2, m − 1), D(n − 1, m − 2)} + c(x n , y m )
        for n in range(2, N):
            for m in range(2, M):
                acm[n, m] = min( acm[n-1, m-1], acm[n-2, m-1], acm[n-1, m-2]) + cm[n, m]

        return acm


    ### optimal warping path owp
    def optimal_warping_path(self, acm, b=None ):
        
        print('\tOptimal warping path (owp)')

        ### determine acm shape
        N, M = acm.shape

        ### if subsequence owp requested
        if b:
            if b < M:
                M = b
            else:
                print('Subsequence length must be below total array length')
        else:
            print('Matching entire query sequence')

        ### move one entry in reverse
        n = N - 1
        m = M - 1

        ### owp to be populated in reverse
        p = []
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
                optimal_step = np.argmin( [ acm[n-1,m-1], acm[n-1,m], acm[n,m-1] ] )
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

        ### B.C.
        p.append([0,0])

        owp = np.asarray( p )
        owp = np.flip( owp )

        return owp


    ### warped sequence
    def warped_sequence(self, sequence, owp):

        warped_sequence = [ [ owp[i,1], sequence[ owp[i,0] ] ] for i in range( owp.shape[0] - 1 ) ]
        warped_sequence = np.asarray( warped_sequence )

        return warped_sequence