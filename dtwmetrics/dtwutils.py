'''
Daniel Vogler
 
Utilities: 
- plotting 

'''

from matplotlib import pyplot as plt
import numpy as np
from dtwmetrics.dtwmetrics import DTWMetrics

dtwm = DTWMetrics()


class DTWUtils:

    def plot_sequences(self, reference, dataset ):

        fig = plt.figure(num=None, figsize=(200, 150), dpi=80, facecolor='w', edgecolor='k')
        p = plt.scatter(reference[:,0],reference[:,1],s=500,marker='.',c='k',label="Reference")
        p = plt.scatter(dataset[:,0],dataset[:,1],s=500,marker='.',c='r',label="Dataset")
        plt.legend(loc='upper center')
        plt.xlabel("Time [-]")
        plt.ylabel("Value [-]")

        return


    def plot_cost_matrix(self, reference, dataset ):
        
        ### cost matrix 
        cm = dtwm.cost_matrix( reference[:,1] , dataset[:,1] )
        ### dtw
        D = dtwm.acm( reference, dataset )
        owp = dtwm.optimal_warping_path( D )

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
        x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax)

        # scatter points on the main axes
        main_ax.pcolormesh(cm)
        main_ax.plot(owp[:,1],owp[:,0],color='w')
        main_ax.yaxis.tick_right()
        main_ax.xaxis.tick_top()
        main_ax.set_title('Cost matrix')

        # plots on the attached axes
        x_plot.plot(np.linspace(0,len(dataset[:,1]),len(dataset[:,1])), dataset[:,1], color='gray')
        x_plot.invert_yaxis()
        x_plot.set_ylim([-1.5,1.5])
        x_plot.set_xlabel('Query [-]')
        # y-axis
        y_plot.plot( reference[:,1] , np.linspace(0,len(reference[:,1]),len(reference[:,1])), color='gray')
        y_plot.invert_xaxis()
        y_plot.set_xlim([1.5,-1.5])
        y_plot.set_ylabel('Reference [-]')

        return