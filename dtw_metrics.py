'''
Daniel Vogler

References: 
(1) Müller, Meinard. Information retrieval for music and motion. Vol. 2. 
    Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3

'''

from matplotlib import pyplot as plt
import numpy as np
from math import pi 


### cost matrix
def cost_matrix(target, estimate):

    ### tile target data
    target_tiled = np.tile(target, (np.size(estimate),1) )
    ### cost matrix
    cost_matrix = np.absolute( ( target_tiled.T - estimate ).T )

    return cost_matrix