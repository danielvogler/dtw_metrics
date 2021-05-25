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


    def plot_warped_sequences(self, reference, query, owp ):

        reference = dtwm.dim_check( reference )
        query = dtwm.dim_check( query )

        fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        font = {'size'   : 14}
        plt.rc('font', **font)
        
        ### reference dim check
        if min( reference.shape ) == 1:
            p = plt.plot(reference,marker='.',c='k',label="Reference", linestyle='None')
        else:
            p = plt.scatter(reference[:,0],reference[:,1],marker='.',c='k',label="Reference")
        
        ### query dim check
        if min( query.shape ) == 1:
            p = plt.plot(query,marker='.',c='r',label="Query", linestyle='None')
        else:
            p = plt.scatter(query[:,0],query[:,1],marker='.',c='r',label="Query")
        
        ### warped sequence
        warped_query = dtwm.warped_sequence(query, owp)
        if min( warped_query.shape ) == 1:
            p = plt.plot(warped_query,marker='.',c='b',label="Warped query", linestyle='None')
        else:
            p = plt.scatter(warped_query[:,0],warped_query[:,1],marker='.',c='b',label="Warped query")
        
        plt.legend(loc='upper center')
        plt.xlabel("Index [-]")
        plt.ylabel("Value [-]")
        plt.title("Sequences")

        return
        

    def plot_matrix(self, reference, query, matrix, owp=None , distance_metric='euclidean' , plot_dim=1, title='Matrix' ):

        reference = dtwm.dim_check( reference )
        query = dtwm.dim_check( query )

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        font = {'size'   : 14}
        plt.rc('font', **font)
        grid = plt.GridSpec(6, 6, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
        x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax)

        main_ax.set_title(title)
        ### plot passed patrix
        main_ax.pcolormesh( matrix )
        main_ax.yaxis.tick_right()
        main_ax.xaxis.tick_top()

        ### plot owp if given
        try:
            main_ax.plot(owp[:,0],owp[:,1],color='w')
        except:
            return

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