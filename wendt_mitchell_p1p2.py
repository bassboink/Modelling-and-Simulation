# -*- coding: utf-8 -*-
"""
wendt_mitchell_p1p2.py
Created on Sun Sep 16 15:08:05 2018
Created by: Mitchell Wendt
Revised: 09/17/2018

People who helped me:
James F. Rathman (base code)
Stackoverflow user wordsforthewise (memoryUse and cpuPercent fetching)
    (https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python)
Stackoverflow user Gaimpaolo Rodola (time.sleep() to reliably generate successive)
    (https://stackoverflow.com/questions/2311301/reliably-monitor-current-cpu-usage)
Stackoverflow user sacul (supressing text output of matplotlib)
    (https://stackoverflow.com/questions/51565320/supress-output-in-matplotlib)
Stackoverflow user ImportanceOfBeingEarnest (for the use of %matplotlib tk)
    (https://stackoverflow.com/questions/25333732/matplotlib-animation-not-working-in-ipython-notebook-blank-plot)
Matplotlib documentation for 3D plotting and rotation
    (https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)
    (https://matplotlib.org/examples/mplot3d/rotate_axes3d_demo.html)
"""

#%matplotlib tk     #This line is needed to be entered into the Spyder IPython console, throws an error in the script. Used for 3d plotting to display the animations in different figures
import numpy as np  #used for the use of arrays, as well as a control method of random number generation
import time         #used to fetch the current Unix time as well as incorporating delays to allow for better random number generation of successive seeds
import os           #used in fetching the system's current memory usage to contribute to more randomness in generating the seeds
import psutil       #used in fetching the system's current cpu and memory usage to contribute more randomness in generating seeds
import matplotlib.pyplot as plt            #used for all 2d plotting purposes, generating scatterplot of darts thrown at the simulated dartboard
from mpl_toolkits.mplot3d import axes3d    #used in generating the 3d plots. Note that Spyder claims that this does not need to be imported, as axes3d is not explicitly called, but the script throws an error if trying to generate the 3d plots without this line

#p1 stuff

#function definition of p1 to generate random numbers (see description in function)
def p1(size = None, method = 'NR', seed = None, returnSeed = False):

    """

    Pseudorandom numbers uniformly distributed [0, 1]
    Uses a linear congruential generator algorithm to
    generate a set of pseudorandom numbers uniformly distributed
    from 0 to 1 for any given size (1D Column array or any dimension tuple).
    By default passes a single value, a generated seed.

    Parameter values (a, b, and c) for the generator depend on the specified
    METHOD, a string variable that must be either 'NR' (Numerical Recipes)
    or 'RANDU' (the RANDU generator). BY default uses 'NR' method

    Seed parameter can be passed if the user wishes to pass a specific seed 
    in order to have repeatable results. By default generates a seed.

    returnSeed parameter allows the seed generated by the program to be used
    outside of the function, if desired. By default doesn't return seed.

    p1 function based largely on James F. Rathman's provided base code.

    """

    #branching section to specify a, b, and c constants based on specified method
    if method == 'NR':   #constants to be set if the user specifies the NR method. Also the default values
        a = 1664525
        b = 1013904223
        c = 2**32
    elif method == 'RANDU':   #constants to be set if the RANDU method is specified
        a = 65539
        b = 0
        c = 2**31
    else:   #if method is not NR or RANDU, system does not have constants for undefined methods
        raise Exception('Error - No such method')   #giving a clear error message to the user that the method provided does not have known constants

    #additional function fetch_cpu_mem defined to determine the system's current cpu and memory usage to contribute to randomness in seed generation
    def fetch_cpu_mem():                             #function passes no parameters, simply returns cpu usage (in percent) and memory usage (in GB)
        pid = os.getpid()                            #obtain the system's current data from the PID controller
        py = psutil.Process(pid)                     #obtain data on current processes from the fetched PID data
        memoryUse = py.memory_info()[0]/2.**30       #calculates memory usage (in GB) (Stackoverflow user wordsforthewise)
        cpuPercent = psutil.cpu_percent()            #python function that fetches 
        return cpuPercent, memoryUse                 #return CPU and memory use values

    #Seed generation, code is not executed if a seed is given.
    if seed == None:

        cpuPercent, memoryUse = fetch_cpu_mem()                               #calculate current CPU and memory usage
        seed = (((time.time() * memoryUse * cpuPercent) % 1) * 1e12) % c      #calculates a seed based on current Unix time, cpu percent, and memory usage. Float value is truncated to its decimal, then expanded. mod c to keep seed within domain value
        time.sleep(0.1)                                                       #delay of 0.1 s after seed is generated to assure that successive seeds do not share the same time, CPU usage, or Memory Usage

        #additional loop to account for cases when CPU usage is 0.0 percent (since my computer's CPU is quite fast, this was a common problem)
        while seed/c == 0:                                                    #keep generating new seeds until the seed is not zero
            time.sleep(1)                                                     #delay of 1 second (the two sets of delays, the default 0.1 s delay and 1 s delay for when the CPU usage is zero seemed to work best on my end to generate seeds quickly)
            cpuPercent, memoryUse = fetch_cpu_mem()                           #reobtain current cpu usage and memory usage
            seed = (((time.time() * memoryUse * cpuPercent) % 1) * 1e12) % c #recalculate the seed with new cpu usage and memory

    #branching code for cases when only one random value is desired
    if size == 1 or size == None: #both cases needed so that by default (and if user specifies a size of 1) a single value NOT inside a list is returned
        if returnSeed:            #different return case if the user requests for the seed to be returned
            return seed/c, seed   #return both the first randomly generated value and the seed value
        else:                     #otherwise just return the first randomly generated value
            return seed/c

    y = np.zeros(size)            #allocate an array for random values based on the user specified size

    #iterate through all indices of specified size. needs to accomodate for any given dimension of array. rather than starting with a flat array and then using np.reshape, this method directly iterates through each index
    firstIter = True                                   #boolean used in determining if the given iteration is the first index. Since no previous knowledge of the array dimension is given, the first index cannot be explicity listed
    for index, val in np.ndenumerate(y):               #iterate through each index of the array, storing both the indices and array value at each iteration, even though the values are currently all set to zero
        if firstIter:                                  #actions taken on the array's first index
            y[index] = seed                            #setting the first index of the array to be the seed value
            prev_index = index                         #prev_index variable defined to store the previous iteration's index, used to reference previous index in LCG algorithm regardless of array shape or dimension
            firstIter = False                          #sets firstIter to be false so that the rest of the values are generated with thenlinear congruential generator algorithm
        else:                                          #done in all other iterations
            y[index] = (a * y[prev_index] + b) % c     #calculate the next pseudorandom value based on the linear congruential generator algorithm and method specified constants, based on the previous index's value
            prev_index = index                         #store the current index as the previous index for the next iteration

    #final return statements (array is divided by c since the LCG algorithm generates values from 0 to c, dividing by c normalizes values to be from 0 to 1)
    if returnSeed:             #done if user specifies to return the seed
        return (y / c), seed   #return the array of random values and the seed
    return y / c               #return the array of random values

#3D scatterplots to verify randomness

#NR Method
fig = plt.figure()                            #set up a new figure to show 3d generated points based on the NR method
ax = fig.add_subplot(111, projection='3d')    #define the 3d axes of the figure

nr_vals = p1((5000,3))                        #calculate the randomized coordinates
x = nr_vals[:,0]                              #extract x values
y = nr_vals[:,1]                              #extract y values
z = nr_vals[:,2]                              #extract z values

for c, m, zlow, zhigh in [('r', 'o', 0, 1)]:  #loop to plot each point on the 3d scatterplot. sets the color and marker
    ax.scatter(x, y, z, c=c, marker=m)        #plotting each set of x, y, and z coordinates

ax.set_title('NR Method Values')              #setting the title of the scatterplot
ax.set_xlabel('X Value')                      #set the x axis label
ax.set_ylabel('Y Value')                      #set the y axis label
ax.set_zlabel('Z Value')                      #set the z axis label

for angle in range(0, 360):                   #iterate from 0 to 360 degrees with a step size of 1 degree
   ax.view_init(30, angle)                    #sets the angle at which the scatterplot is shown
   plt.draw()                                 #display the scatterplot at the current angle
   plt.pause(.001)                            #pause for 0.001 s between angles to set the speed of the animation. Cannot be replaced by time.sleep()

#RANDU Method
fig1 = plt.figure()                           #set up a new figure to show 3d generated points based on the RANDU method
ax = fig1.add_subplot(111, projection='3d')   #define the 3d axes of the figure

randu_vals = p1((5000,3), method = 'RANDU')   #calculate the randomized coordinates
x_r = randu_vals[:,0]                         #extract x values
y_r = randu_vals[:,1]                         #extract y values
z_r = randu_vals[:,2]                         #extract z values

for c, m, zlow, zhigh in [('b', 'o', 0, 1)]:  #loop to plot each point on the 3d scatterplot. sets the color and marker
    ax.scatter(x_r, y_r, z_r, c=c, marker=m)  #plotting each set of x, y, and z coordinates

ax.set_title('RANDU Method Values')           #setting the title of the scatterplot
ax.set_xlabel('X Value')                      #set the x axis label
ax.set_ylabel('Y Value')                      #set the y axis label
ax.set_zlabel('Z Value')                      #set the z axis label

for angle in range(0, 360):                   #iterate from 0 to 360 degrees with a step size of 1 degree
   ax.view_init(30, angle)                    #sets the angle at which the scatterplot is shown
   plt.draw()                                 #display the scatterplot at the current angle
   plt.pause(.001)                            #pause for 0.001 s between angles to set the speed of the animation. Cannot be replaced by time.sleep()

#Native Python Method
fig2 = plt.figure()                           #set up a new figure to show 3d generated points based on the native python method
ax = fig2.add_subplot(111, projection='3d')   #define the 3d axes of the figure

native_vals = np.random.rand(5000,3)          #calculate the randomized coordinates
x_n = native_vals[:,0]                        #extract x values
y_n = native_vals[:,1]                        #extract y values
z_n = native_vals[:,2]                        #extract z values

for c, m, zlow, zhigh in [('g', 'o', 0, 1)]:  #loop to plot each point on the 3d scatterplot. sets the color and marker
    ax.scatter(x_n, y_n, z_n, c=c, marker=m)  #plotting each set of x, y, and z coordinates

ax.set_title('Native Python Method Values')   #setting the title of the scatterplot
ax.set_xlabel('X Value')                      #set the x axis label
ax.set_ylabel('Y Value')                      #set the y axis label
ax.set_zlabel('Z Value')                      #set the z axis label

for angle in range(0, 360):                   #iterate from 0 to 360 degrees with a step size of 1 degree
   ax.view_init(30, angle)                    #sets the angle at which the scatterplot is shown
   plt.draw()                                 #display the scatterplot at the current angle
   plt.pause(.001)                            #pause for 0.001 s between angles to set the speed of the animation. Cannot be replaced by time.sleep()



#Extra bit of code: Verify randomness of consecutively generated seeds in 2D plot. Commented out as this process takes a bit of time, can be uncommented
'''
seeds = np.zeros(100)                                    #preallocate an array of seed values
num = np.zeros(100)                                      #preallocate an array of index values of each seed value
for i in range(0,100):                                   #generate seeds 100 times
    seeds[i] = p1(1)                                     #generate a single randomly generated seed using the NR constants
    num[i] = i                                           #fetch the current index in num

#plotting the seeds to visually analyze the randomness of seeds. could be made 3d as in other previous scatterplots, but takes time due to the delay between seeds
plt.plot(num, seeds)                                     #plot the seed values versus index value
plt.title('Verifying Randomness of Successive Seeds')    #set the title of the plot
plt.xlabel('seed generated')                             #setting the x axis title
plt.ylabel('value')                                      #setting the y axis title
plt.show()                                               #display the plot
'''

#test function calls from homework guidelines
print(p1())                                              #generate a single random seed
print(p1(5))                                             #generate 5 random values using the NR method
print(p1((3,20), 'RANDU'))                               #generate a 3 x 20 array of random values using the RANDU method
print(p1(200, returnSeed = True))                        #generate 200 random values using the NR method and return the seed
print(p1(method = 'RANDU', size = (4, 4)))               #generate a 4 x 4 array of random values using the RANDU method

#p2 stuff

#function definition of p2 to simulate dart throwing to estimate pi (see description in function)
def p2(nThrows = 200, method = 'NR'):

    """

    Simulates throwing darts at a square dartboard with an inscribed
    circle with center (0.5,0.5), radius 0.5, and side length 1

    Each dart is given a randomly generated coordinate in the x and y direction
    and is then plotted in a 2d scatterplot that represents the dartboard

    nThrows parameter defines the number of darts that are thrown at the dartboard.
    By default uses 200 throws.

    method parameter is the same as in p1, defines which constants are used in
    the LCG algorithm. Uses NR method by default.

    """

    num_in_circle = 0                      #counter that determines the number of darts that hit the inside of the circle
    x = p1(nThrows, method)                #randomly generates x coordinates
    y = p1(nThrows, method)                #randomly generates y coordinates
    in_circle = []                         #empty list to be filled with indices at which dart throws hit inside the circle
    outside_circle = []                    #empty list to be filled with the indices at which dart throws hit outside the circle
    for i in range(0,nThrows):             #for each throw iterate to see if the dart hit inside or outside the circle
        if np.sqrt((np.absolute(x[i] - 0.5))**2 + (np.absolute(y[i] - 0.5))**2) < 0.5:  #if the radius of the dart throw is less than the radius of the circle
            num_in_circle += 1             #increment the counter of darts thrown inside the circle
            in_circle.append(i)            #add the current index to the list of indices that the dart hit inside the circle
        else:                              #if the dart hits outside the circle
            outside_circle.append(i)       #add the current index to the list of indices that the dart hit outside the circle

    pi_est = 4.0*num_in_circle/nThrows     #calculate an estimate of pi based on the total number of throws and the number of throws inside the circle.

    """
    Note: this estimation method works because the area of the square is 1 x 1
    and the area of the circle is pi*r ^2, or pi*(0.5)^2 which is equal to pi/4
    This means the ratio between the area of the circle and the area of the 
    square is pi/4. This is the case regardless of what the radius of the circle
    is set to be. The number of darts thrown in each region can be considered 
    to be exactly proportional to the area of each region as long as the darts
    are being thrown in random coordinates.
    """

    fig3 = plt.figure()                                                                       #generate a new plot to show the 2d scatterplot that represents the dartboard
    ax1 = fig3.add_subplot(111)                                                               #sed up the axes of the new plot

    ax1.scatter(x[in_circle], y[in_circle], c = 'r', label = 'in circle')                     #plot the points inside the circle as red points
    ax1.scatter(x[outside_circle], y[outside_circle], c = 'b', label = 'outside circle')      #plot the points outside the circle as blue points
    plt.title('Positions of randomized throws at dartboard, pi = ' + str(pi_est))             #sets the title of the scatterplot and shows an estimate of pi based on the simulation
    plt.xlabel('x coordinate')                                                                #set the x axis title
    plt.ylabel('y coordinate')                                                                #set the y axis title
    plt.legend()                                                                              #display a legend on the scatterplot
    plt.show()                                                                                #show the plot of the simulated dartboard

    return pi_est                                                                             #return the estimated pi value each time the function is called

#Generate the final estimate of pi
pi_val = p2()                                                                                 #fetch an estimate for pi for a default simulation of 200 throws
print("pi = " + str(pi_val))                                                                  #display the final estimated value for pi





