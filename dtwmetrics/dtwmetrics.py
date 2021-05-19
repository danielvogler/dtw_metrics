'''
Daniel Vogler

References: 
(1) Müller, Meinard. Information retrieval for music and motion. Vol. 2. 
    Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3

'''

import numpy as np
from matplotlib import pyplot as plt

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


    def plot_sequences(self, reference, dataset ):

        fig = plt.figure(num=None, figsize=(200, 150), dpi=80, facecolor='w', edgecolor='k')
        p = plt.scatter(reference[0],reference[1],s=500,marker='.',c='k',label="Reference")
        p = plt.scatter(dataset[0],dataset[1],s=500,marker='.',c='r',label="Dataset")
        plt.legend(loc='upper center')
        plt.xlabel("Time [-]")
        plt.ylabel("Value [-]")

        return


    def plot_cost_matrix(self, reference, dataset ):

        x_1 = reference[0]
        y_1 = reference[1]
        x_2 = dataset[0]
        y_2 = dataset[1]
        
        ### cost matrix 
        cm = self.cost_matrix(y_1,y_2)
        ### dtw
        D = self.acm( [y_1,x_1], [y_2,x_2] )
        owp = self.optimal_warping_path( D )

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
        x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax)

        # scatter points on the main axes
        main_ax.pcolormesh(cm)
        main_ax.plot(owp[:,0],owp[:,1],color='w')
        main_ax.yaxis.tick_right()
        main_ax.xaxis.tick_top()
        main_ax.set_title('Cost matrix')

        # plots on the attached axes
        x_plot.plot(np.linspace(0,len(y_2),len(y_2)), y_2, color='gray')
        x_plot.invert_yaxis()
        x_plot.set_ylim([-1.5,1.5])
        x_plot.set_xlabel('Query [-]')
        # y-axis
        y_plot.plot(y_1, np.linspace(0,len(y_1),len(y_1)), color='gray')
        y_plot.invert_xaxis()
        y_plot.set_xlim([1.5,-1.5])
        y_plot.set_ylabel('Reference [-]')

        return