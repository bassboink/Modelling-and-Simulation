"""
final_project.py
Created on Sun Dec 9 11:12:34 2018
Created by: Mitchell Wendt
Revised: 12/11/2018

People who helped me: The Wikipedia Community (for discussion about the Prim algorithm, base 
                                               code for generating the seed maze
                                               https://en.wikipedia.org/wiki/Maze_generation_algorithm)
                      Documentation of matplotlib, but also not really because it's not easy to use
                      Ryan Arnold (for help with yanking my hair out over the matplotlib animation
                                   module, getting the prerendered animation to work with passing the
                                   previous move into the move function)
                      Dr. Jim Rathman (for helpful insights of random walk simulations, cellular
                                       automata, and overall information about creating python simulations)
"""

###IMPORTANT: Read the docstring of the move() function to learn more about the performance behavior of the program

import numpy as np                          #import numpy as np for random number generation and use of arrays
import matplotlib.pyplot as plt             #import pyplot for use of figure plotting
import matplotlib.animation as animation    #import animation for the animation of the plot window

def myfinal(width=25, height=25, density=.75, complexity=.75, onesol = True, animate = True):
    
    '''
        
    This function generates a maze and moves a mouse throught the maze until 
    the maze is solved, and animates the movement of the mouse in
    preallocated image frames (impacts performance, see docstring of the
    move() function for more on this). This function takes 6 inputs:
    width: number of x coordinates in the maze
    height: number of y coordinates in the maze
    density: parameter that determines how full the final maze is 
             (note: not the exact density, since the maze density is
             mathematically bounded due to the fact that empty space
             must surround each wall on both sides). Also known as the number
             of islands.
    complexity: parameter that determines how much branching occurs from each 
                island
    onesol: flag that determines if borders should be filled in and 
            one maze exit be created or not
    animate: flag that determines if the animation window should be dispalyed
             or not. 
    
    This function returns the animation object created by the animation module
    if animation is present (for use in saving video, etc.) and returns the 
    number of generations whenever animation is not used (for quick 
    calculation needed moves of successive runs if desired)
                    
    '''
    
    #throw error if non integers are passed to height and width
    if type(height) is not int or type(width) is not int:
        raise TypeError("height and width must be positive integers")
    
    #throw an error if height or width are less than 5 (3 is trivial, 4 is not possible since odd/even is specified to be odd)
    elif height < 5 or width < 5:
        raise ValueError("height and width must be positive integers greater than or equal to 5")
    
    #throw an error if onesol or animate flags are not booleans
    if type(onesol) is not bool or type(animate) is not bool:
        raise TypeError("animate and onesol paramaters must be bool")
    
    #throw an error if complexity or density values are not floats
    if type(complexity) is not float or type(density) is not float:
        raise TypeError("complexity and density must be float values")
    
    #throw an error if negative values are passed to density or complexity
    elif complexity < 0 or density < 0:
        raise ValueError("complexity and density must be greater than zero")
    
    def gen_maze(width, height, complexity, density, onesol):
        
        '''
        
        This function generates the initial maze inside the Z matrix. This 
        function takes 5 inputs:
            width: number of x coordinates in the maze
            height: number of y coordinates in the maze
            complexity: parameter that determines how much branching occurs
            density: parameter that determines how full the final maze is 
                     (note: not the exact density, since the maze density is
                     mathematically bounded due to the fact that empty space
                     must surround each wall on both sides). Number of islands.
            onesol: flag that determines if borders should be filled in and 
                    one maze exit be created or not
                
        This function returns the Z array that stores all the information about
        the empty space (represented by 0), walls (1), and mouse (2) are. The
        maze is generated from a variation of the Prim algorithm, where rather
        than the array being filled with walls at the start the maze is empty
        at the start. Walls of length 2 are started at random locations, called
        islands. This allows for easy iteration over python arrays. 
        
        '''
        
        #define the complexity and density values, scaled based on the size of the maze
        complexity = int(complexity * (5 * (height + width)))
        density    = int(density * ((height // 2) * (width // 2)))
        
        #preallocate the array that stores current information about free space (0), walls(1), and the mouse (2)
        Z = np.zeros([height, width], dtype=int)
        
        #if only one solution is desired, set the borders of the array to be walls (makes for harder, prettier mazes)
        if onesol:
            Z[0, :] = Z[-1, :] = 1
            Z[:, 0] = Z[:, -1] = 1
            
        #iterate through the density value
        for i in range(density):
            
            #generate islands at random coordinates, placing a wall where the island is
            x = np.random.randint(0, width // 2) * 2
            y = np.random.randint(0, height // 2) * 2
            Z[y, x] = 1
            
            #iterate through the complexity value
            for j in range(complexity):
                
                #calculate a neighborhood of the coordinates that are 2 units north, south, east, and west of current point
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < width - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < height - 2:  neighbours.append((y + 2, x))
                
                #whenever there are neighbors, choose a random neighbor
                if len(neighbours):
                    y_,x_ = neighbours[np.random.randint(0, len(neighbours) - 1)]
                    
                    #whenever the randomly selected neighbor is empty, make it a wall as well as the space
                    #in between the current point and the selected neighbor (makes a wall of length 2, lkeeps branching)
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        
                        #change the current coorinates to be the randomly selected neighbor
                        x, y = x_, y_
                        
        #set the mouse to be in the center of the maze
        Z[(height // 2), (width // 2)] = 2
        
        #if only one solution is in the maze, make a single space to be an empty space that is the exit
        if onesol:
            
            #generate a random float from 0 to 1, and define an index variable
            u = np.random.rand()
            v = 0
            
            #1/4 chance of generating the exit on the north edge
            if u < 0.25:
                
                #generate potential exit points until one is not at a branching point (assures the exit is usable)
                while Z[v,1] != 0:
                    v = np.random.randint(1,height-2)
                
                #create the exit
                Z[v,0] = 0
            
            #1/4 chance of generating the exit on the south edge
            elif u < 0.5: 
                
                #generate potential exit points until one is not at a branching point (assures the exit is usable)
                while Z[v,(width-2)] != 0:
                    v = np.random.randint(1,(height-2))
                
                #create the exit
                Z[v,(width-1)] = 0
            
            #1/4 chance of generating the exit on the east edge
            elif u < 0.75:
                
                #generate potential exit points until one is not at a branching point (assures the exit is usable)
                while Z[1,v] != 0:
                    v = np.random.randint(1,width-2)
                
                #create the exit
                Z[0,v] = 0
            
            #1/4 chance of generating the exit on the west edge
            else: 
                
                #generate potential exit points until one is not at a branching point (assures the exit is usable)
                while Z[(height-2),v] != 0:
                    v = np.random.randint(1,(width-2))
                
                #create the exit
                Z[(height-1),v] = 0
        
        #return the Z array
        return Z
    
    def move(prev_dir):
        
        '''
        
        This function moves the mouse from one space to the next at each move. 
        This function takes one paramter as an input (the previous direction
        that the mouse just moved) and returns two outputs (the Z array that 
        stores the information about the mouse and wall locations and the 
        direction that the mouse just moved)
        
        It is important to talk about why the previous direction is needed.
        When this function was first made, the mouse would simply move
        randomly. While functional, this was a terrible strategy and took the 
        mouse an aggravatingly long time to solve the maze, even in easy 
        situations. Choosing to not go the previous direction allows for the 
        mouse to backtrack less and solve a bit faster.
        
        However, this also shaped the way that I had to animate the maze. I
        initially used the FuncAnimation() function in the animation package to
        animate, and it performed very quickly as generations were made on the 
        fly and little memory was needed. However, the fargs() functionality
        did not quite work correctly and the previous direction could not
        feasibly be passed into the move function. ArtistAnimation() is used 
        instead, which preallocates all frames at the start. While this allows
        for the number of total moves made to be fast to calculate (< 1 sec),
        this makes animation take a while. Clicking the figure window too early 
        has led to crases on my end, so DO NOT DO THIS! The animation loops so 
        the whole animation can be viewed regardless of when the window is 
        opened by the user once it is loaded.
        
        '''
        
        #fetch the x and y coordinates of where the mouse currently is
        ismouse = np.isin(Z, 2)
        indices = np.where(ismouse)
        xcoord = indices[0][0]
        ycoord = indices[1][0]
        
        #define a neighbor counter and list that stores possible directions to move
        neighbors = 0
        directions = []
        
        #add the east neighbor if it is empty
        if Z[xcoord+1, ycoord] == 0:
            neighbors += 1
            directions.append('e')
        
        #add the west neighbor if it is empty
        if Z[xcoord-1, ycoord] == 0:
            neighbors += 1
            directions.append('w')
        
        #add the south neighbor if it is empty
        if Z[xcoord, ycoord+1] == 0:
            neighbors += 1
            directions.append('s')
        
        #add the north neighbor if it is empty
        if Z[xcoord, ycoord-1] == 0:
            neighbors += 1
            directions.append('n')
        
        #in the rare case where the mouse is completely surrounded by walls, choose to go north, which
        #will automatically break the north wall and allow the mouse to escape
        if neighbors == 0:
            new_dir = 'n'
        
        #whenever there is only one direction that can be moved, move in that direction
        elif neighbors == 1:
            new_dir = directions[0]
        
        #otherwise, pick a random direction that is not the opposite of the last direction (don't backtrack if possible)
        else:
            
            #remove the opposite of the last direction from the list of potential directions
            if prev_dir in directions:
                if prev_dir == 'n':
                    directions.remove('s')
                elif prev_dir == 's':
                    directions.remove('n')
                elif prev_dir == 'e':
                    directions.remove('w')
                else:
                    directions.remove('e')
            
            #pick a random direction to be the new direction
            w = np.random.randint(0,len(directions))
            new_dir = directions[w]
        
        #move east, vacating the previous space and occupying the space east
        if new_dir == 'e':
            Z[xcoord,ycoord] = 0
            Z[xcoord+1,ycoord] = 2
        
        #move west, vacating the previous space and occupying the space west
        elif new_dir == 'w':
            Z[xcoord,ycoord] = 0
            Z[xcoord-1,ycoord] = 2
        
        #move south, vacating the previous space and occupying the space south
        elif new_dir == 's':
            Z[xcoord,ycoord] = 0
            Z[xcoord,ycoord+1] = 2
        
        #move north, vacating the previous space and occupying the space north
        else:
            Z[xcoord,ycoord] = 0
            Z[xcoord,ycoord-1] = 2
        
        #return the updated Z array and the direction that the mouse just moved
        return Z, new_dir
    
    #define a plot figure if it is set to animate
    if animate:
        fig = plt.figure(figsize=(5, 5))
    
    #make the height and width to be odd numbers only. makes for better looking mazes
    height = (height // 2) * 2 + 1
    width = (width // 2) * 2 + 1
    Z = gen_maze(width, height, complexity, density, onesol)
    prev_dir = ''
    
    #iterate through moves until the maze is solved by the mouse
    
    #define a generation counter and list to store image frames
    n = 0 
    frames = []
    
    #loop until a break statement is encountered
    while True:  
        
        #call move() to get the new direction and location of the mouse
        Z, prev_dir = move(prev_dir)
        
        #if animating, create a plot and put in the image list
        if animate:
            cplot = plt.imshow(Z, cmap=plt.cm.binary, interpolation='nearest')
            frames.append([cplot])
        
        #if the maze is solved (mouse is at any border of the maze), stop running!
        if ((Z[:,0] == 2).any() or (Z[0,:] == 2).any() or \
            (Z[height-1,:] == 2).any() or (Z[:,width-1] == 2).any()):
            break
        
        #otherwise, just move onto the next generation
        else:
            n+=1

    #if animating, set the title of the animation, axes titles, and animate!
    if animate:
        plt.title("Mouse Traversing the Maze: Takes "+str(n)+" Moves")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay = 1000)
        
        #return the animation object if desired to be saved, etc. (also keeps Spyder from giving a warning for an unused 'ani')
        return ani
    
    #otherwise, return the number of generations
    else:
        return n
    

        
