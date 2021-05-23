'''
Daniel Vogler
 
Utilities: 
- plotting 

'''

from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from dtwmetrics.dtwmetrics import DTWMetrics

dtwm = DTWMetrics()


class DTWUtils:

    def plot_sequences(self, reference, query ):

        reference = dtwm.dim_check( reference )
        query = dtwm.dim_check( query )

        fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        font = {'size'   : 14}
        plt.rc('font', **font)
        
        ### reference dim check
        if min( reference.shape ) == 1:
            p = plt.plot(reference,marker='.',c='k',label="Reference")
        else:
            p = plt.scatter(reference[:,0],reference[:,1],s=500,marker='.',c='k',label="Reference")
        
        ### query dim check
        if min( query.shape ) == 1:
            p = plt.plot(query,marker='.',c='r',label="Query")
        else:
            p = plt.scatter(query[:,0],query[:,1],s=500,marker='.',c='r',label="Query")
        plt.legend(loc='upper center')
        plt.xlabel("Time [-]")
        plt.ylabel("Value [-]")
        plt.title("Time sequence")

        return




    def plot_matrix(self, reference, query, distance_metric='euclidean' , plot_dim=1, matrix='cost' ):

        ### cost matrix 
        #cm = dtwm.cost_matrix(reference, query)
        cm = cdist(reference, query, metric=distance_metric)
        ### dtw
        acm = dtwm.acm( reference, query )
        owp = dtwm.optimal_warping_path( acm )

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        font = {'size'   : 14}
        plt.rc('font', **font)
        grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
        x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax)

        # scatter points on the main axes
        if matrix == 'cm':
            main_ax.pcolormesh( cm )
            main_ax.set_title('Cost matrix')
        elif matrix == 'acm':
            main_ax.pcolormesh( acm )
            main_ax.set_title('Accumulated cost matrix')

        main_ax.plot(owp[:,0],owp[:,1],color='w')
        main_ax.yaxis.tick_right()
        main_ax.xaxis.tick_top()

        # plots on the attached axes
        x_plot.plot(np.linspace(0,len(query[:,plot_dim]),len(query[:,plot_dim])), query[:,plot_dim], color='gray')
        x_plot.invert_yaxis()
        x_plot.set_ylim([-1.5,1.5])
        x_plot.set_xlabel('Query [-]')
        # y-axis
        y_plot.plot( reference[:,plot_dim] , np.linspace(0,len(reference[:,plot_dim]),len(reference[:,plot_dim])), color='gray')
        y_plot.invert_xaxis()
        y_plot.set_xlim([1.5,-1.5])
        y_plot.set_ylabel('Reference [-]')

        return