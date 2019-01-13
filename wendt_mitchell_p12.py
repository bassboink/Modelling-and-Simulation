# -*- coding: utf-8 -*-
"""
wendt_mitchell_p12.py
Created on Tue Dec 4 13:18:34 2018
Created by: Mitchell Wendt
Revised: 12/5/2018

People who helped me: Jim Rathman (for providing the forest fire python base file)
                      Ryan Arnold (for help in fixing the ndimage inputs to be
                      numerical rather than boolean)
"""

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def borebabybore(density = 0.6, neighborhood = 'vonNeumann', nGen = None, 
         pbc = False, grid = True):
    
    '''
    
    This function simulates the spread of a bore infestation through a forest
    via a cellular automata model. This function has 5 input parameters:
        
        density: the percent of sites in the forest that contain trees (not bare)
        neighborhood: the type of neighborhood that defines the number of 
                      neighbors that each cell has
        nGen: number of generations to run. If None, simply runs until there are
              no healthy trees or there are no healthy trees with infested neighbors
        pbc: a boolean that determines the boundary condition. Periodic if true,
             constant/dead space if false
        grid: a boolean that determines whether to display gridlines or not
    
    '''

    #Error handling and definition of the types of neighborhoods
    
    #always using a radius of 1 (could make a parameter but assignment said not to)
    radius = 1    
    
    #define Moore neighborhoods of radius 1 and 2
    if neighborhood == 'Moore':
        if radius == 1:
            mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        elif radius == 2:
            mask = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 0, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],])
        
        #throw a value error if radius is not 1 or 2
        else:
            raise ValueError('radius must be 1 or 2')
            
    #define Von Neumann neighborhoods of radius 1 and 2    
    elif neighborhood == 'vonNeumann':
        if radius == 1:
            mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        elif radius == 2:
            mask = np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 0, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0],])
        
        #throw a value error if radius is not 1 or 2
        else:
            raise ValueError('radius must be 1 or 2')
            
    #throw an error if the neighborhood passed in is not either a moore or van neumann neighborhood
    else:
        raise ValueError("neighborhood must be 'Moore' or 'vonNeumann'")
        
    #Other error handling
    
    #throw error if density value is not between 0 or 1
    if type(density) is not float:
        raise TypeError('density value must be a float between 0 and 1')
    elif density < 0 or density > 1:
        raise ValueError('density value must be a float between 0 and 1')
        
    #throw error if nGen is not a positive integer or None
    if type(nGen) is not int and nGen is not None:
        raise TypeError('nGen must either be None or a positive integer')
    elif nGen is not None:
        if nGen < 1: 
            raise ValueError('nGen must either be None or a positive integer')
    
    #throw an error is pbc or grid are not booleans
    if type(pbc) is not bool or type(grid) is not bool:
        raise TypeError('pbc and grid parameters must be booleans')
    
    #periodic boundary conditions
    if pbc:
        bc_mode = 'wrap' 
    
    #deadzone boundary conditions
    else:
        bc_mode = 'constant' 

    #set the dimensions of the simulation space (by number of sites/rows/columns)
    rows = 70
    columns = 140
    
    #initialize all sites to be bare spots (without trees) for now
    z = np.zeros((rows, columns), dtype = int)   
    
    #seed the forest at specified density
    z = np.random.binomial(1, density, (rows, columns))
    
    #set all trees in the center 5x5 box to be infested if there is a tree in the spot
    k = z[int(rows/2-2):int(rows/2+3), int(columns/2-2):int(columns/2+3)] == 1 #find green trees
    z[int(rows/2-2):int(rows/2+3), int(columns/2-2):int(columns/2+3)][k] = 2 #burn, baby, burn
        
    """
    Define colormap to use. Can pick from the many built-in colormaps, or,
    as shown below, create our own. First create a dictionary object with
    keys for red, green, blue. The value for each pair is a tuple of tuples.
    Must have at least 2 tuples per color, but can as many as you wish. The
    first element in each tuple is the position on the colormap, ranging from
    0 (bottom) to 1 (top). The second element is the brightness (gamma) of the
    color. The third element is not used when we only have two tuples per color.
    The conventional red-green-blue (RGB) color scale has gamma values ranging
    from 0 to 255 (256 total levels); these are normalized 0 to 1. I.e., a gamma
    of 1 in the color tuple denotes gamma 255.
    
    The code below creates a color map with
        position = 0.0   white (255, 255, 255) => (1, 1, 1)
        position = 0.33  green (0, 204, 0) => (0, 0.8, 0)
        position = 0.67  orange (255, 102, 0) => (1, 0.4, 0)
        position = 1.0   grey (120, 120, 120) => (0.47, 0.47, 0.47)
    """
    cdict = {'red':   ((0.00, 1.00, 1.00),
                       (0.33, 0.00, 0.00),
                       (0.67, 1.00, 1.00),
                       (1.00, 0.47, 0.47)),
             'green': ((0.00, 1.00, 1.00),
                       (0.33, 0.80, 0.80),
                       (0.67, 0.40, 0.40),
                       (1.00, 0.47, 0.47)),
             'blue':  ((0.00, 1.00, 1.00),
                       (0.33, 0.00, 0.00),
                       (0.67, 0.00, 0.00),
                       (1.00, 0.47, 0.47))}
    
    #now create the colormap object
    colormap = colors.LinearSegmentedColormap('mycolors', cdict, 256)
    
    #set up plot object    
    fig, ax = plt.subplots()
    plt.axis('scaled')
    
    #set dimensions of the simulation window
    plt.axis([0, columns, 0, rows])
    
    #use the colormap as defined above
    cplot = plt.pcolormesh(z, cmap = colormap, vmin = 0, vmax = 3)
    
    #initial title of simulation window at gen = 0
    plt.title('Bore infestation with initial density = ' + str(density) + '\nGeneration 0')  
    
    #display the grid lines if the input parameter grid is True
    if grid:    
        #Adding gridlines seems a bit more complicated than it should be...
        plt.grid(True, which = 'both', color = '0.5', linestyle = '-')
        plt.minorticks_on()
        xminorLocator = ticker.MultipleLocator(1)
        yminorLocator = ticker.MultipleLocator(1)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

    #Define update function to be used to animate the plot
    def update(i):      
        
        '''
        
        This function updates the status of each site at each generation based
        on the defined cellular automata rules listed in the guidelines for
        P12. This function takes one input: i, the current generation number.
        The function outputs cplot, an array containint the updated statuses
        at each site.
        
        '''
        
        #find the number of neighboring infested trees at each site
        nInfest = ndimage.generic_filter(1.0*(z==2), np.sum, footprint = mask, mode = bc_mode)
        
        #calculate probability of becoming infested at each site based on the number of neighbors
        p = nInfest/10.0
              
        #Rules rule!
        """
        States: bare (0), green tree (1), infested tree (2)
        Rules:
          rule 1: green tree with one or more infested neighbors has a probability of n/10 to become infested where n is the number of neighbors
          rule 2: bare spots and infested trees never change
        """
        
        #rule 1
        r1 = (z == 1) & (np.random.binomial(1,p)==1)        
        z[r1] = 2
        
        #rule 2: nothing needs to be added!
        
        #set up plot window and make z array 1D for use in set_array
        cplot.set_array(z.ravel()) #set_array requires a 1D array (no idea why...)
        plt.title('Bore infestation with initial density = ' + str(density) + '\nGeneration ' + str(i))
        plt.xlabel('x coordinate (site number)')
        plt.ylabel('y coordinate (site number)')
        
        return cplot

    """
    We don't know beforehand how many steps (generations) will be required until
    no more trees are burning, so we can't simply set the "frames" parameter in
    FuncAnimation to an integer value. This parameter can also be an iterable
    or a generator; in fact, when used this way it can be used (as done below)
    to decide when to stop or can be used to generate data to be used in the
    update function. The generator below returns integers 0, 1, 2, ... until
    no more trees are burning. The return statement inside a generator indicates
    the generator is finished (no more items), so the animation comes to a 
    graceful stop. Cool!
    """    
    
    def genner():
        
        '''
        
        This function is a generation number generator. It takes no inputs and 
        returns i if the simulation is to continue and nothing if the simulation
        is to be finished. New generations will be generated as long as there 
        are still healthy trees and there are healthy trees that have infested
        neighbors.
        
        '''
        
        #define a generation counter
        i = 1
        
        #iterate through generations until a return statement is encountered
        while True:
            
            #yield the generation counter and increment the generation counter as long as there are healthy trees with infested neighbors
            if (z == 2).any(): #if any trees are burning
                nInfest = ndimage.generic_filter(1.0*(z==2), np.sum, footprint = mask, mode = bc_mode)
                if ((z == 1) & (nInfest > 0)).any():
                    yield i
                    i += 1
            
            #stop generating new generations once the above conditions are not met
            else:
                return    
    
    #base the  number of generations to be run based off of the above criteria whenever nGen is none (does a fixed number of generations otherwise)
    if nGen is None:    
        nGen = genner()
    
    #Time to put those midi-chlorians to work, Ani... setting up the animation object in the figure window.  
    anakin = animation.FuncAnimation(fig, update,
                           frames = nGen, 
                           fargs = (),
                           #init_func = initialize,
                           interval = 400, 
                           blit = False, #cplot is not iterable, so blit must be False
                           repeat = False)
    
    #return the animation object
    return anakin