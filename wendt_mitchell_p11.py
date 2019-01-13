# -*- coding: utf-8 -*-
"""
wendt_mitchell_p11.py
Created on Mon Dec 3 17:34:26 2018
Created by: Mitchell Wendt
Revised: 12/5/2018

People who helped me: Jim Rathman (for providing the drunkard python base file)
"""

#import statements
import numpy as np #numpy for the use of functions such as sin, cos as well as random number generation
import matplotlib.pyplot as plt #matplotlib for displaying the animation of the robovac

#Object definition of the robovac
class robovac:
    
    '''
    
    This is an object definition for a robovac object. The object has 5
    attributes: 
        xpos - the current x coordinate (in feet)
        ypos - the current y coordinate (in feet)
        theta - the current angle (in radians) of which the vacuum is currently 
                moving
        batteryLife - the current batteryLife remaining (in minutes)
        orig_batteryLIfe - the battery capacity of the vacuum (in minutes).
                           this is used to vary the size of the timestep so 
                           that the simulation always runs for one minute
                           (at 30 fps)
        
    The object also contains one method: move(). This method updates the theta
    and coordinates of the vacuum and decreases the current battery life at 
    each timestep
    
    '''
    
    #Constructor method for creating a robovac object
    def __init__(self, start_pos, theta, batteryLife):
        
        #Attributes: an x position, y positing, angle theta of current movement, current battery life and original battery life when full
        self.xpos = start_pos[0]
        self.ypos = start_pos[1]
        self.theta = theta
        self.batteryLife = batteryLife
        self.orig_batteryLife = batteryLife #original battery life is used as it is used to scale the timestep (all sims are @ 30 fps, 1 min)
        
    
    #move method to change the position and direction of the robovac when necessary
    def move(self):
        
        #do not move anymore if the robovac is out of battery
        if self.batteryLife <= 0.0:
            return False
        
        #what to do when the robovac has approached one of the walls or corners...
        elif self.xpos >= 9.5 or self.xpos <= -9.5 or self.ypos >= 9.5 or self.ypos <= -9.5:
            
            #initialize a random direction variable and temporary x and y coordinates
            random_direction = 0.0
            temp_x = self.xpos
            temp_y = self.ypos
            
            #generate temporary movements in a random direction until the robovac is moved away from the wall
            while temp_x >= 9.5 or temp_x <= -9.5 or temp_y >= 9.5 or temp_y <= -9.5:
                
                #generate a random angle from 0 to 2pi
                random_direction = (2*np.random.rand()) * np.pi
                
                #update the positions based on the current timestep (orig_batterylife/30) and the speed of 0.5 ft/s
                temp_x = 0.5 * np.cos(random_direction)*self.orig_batteryLife/30 + self.xpos
                temp_y = 0.5 * np.sin(random_direction)*self.orig_batteryLife/30 + self.ypos
            
            #finally update the coordinates and theta of the robovac
            self.xpos = temp_x
            self.ypos = temp_y
            self.theta = random_direction
            
            #decrement the battery life based on the current timestep (orig_batterylife/30)
            self.batteryLife = self.batteryLife - 1.0/60*self.orig_batteryLife/30
            
            #return True to indicate to keep moving
            return True
        
        #otherwise, just move forward in the same direction the vacuum is already moving and decrement battery life based on timestep (orig_batterylife/30)
        else:
            self.xpos = 0.5 * np.cos(self.theta)*self.orig_batteryLife/30 + self.xpos
            self.ypos = 0.5 * np.sin(self.theta)*self.orig_batteryLife/30 + self.ypos
            self.batteryLife = self.batteryLife - 1.0/60*self.orig_batteryLife/30
            
            #return True to indicate to keep moving
            return True

#Main program
def clean(position = (0.0, 0.0), theta = np.pi/4, batteryLife = 30.0):
    
    '''
    
    This is the main function of the program. This program creates an instance
    of a robovac object and animates its motion across a 20 ft x 20 ft room.
    The function takes three input parameters:
        position: a tuple of two floats that define the initial position of the
                  robovac (in feet)
        theta: the initial direction of movement of the vacuum (in radians)
        batteryLife: the initial battery life of the vacuum (in minutes)
        
    The function does not return anything
    '''
    
    #define a robovac roomba based on the passed in parameters
    roomba = robovac(position, theta, batteryLife)
    
    #set up the animation window
    plt.figure()
    ax = plt.axes()
    
    #set up a movable point with the correct size to represent the roomba
    point, = ax.plot([], [], 'ko',markersize = 15)
    
    #set the x and y bounds of the room as per the P11 guidelines
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    #make the aspect ratio of the animation window to be perfectly square
    plt.gca().set_aspect('equal', adjustable='box')
    
    #titles and axes labels of the animation window
    plt.title('Robovac Locations')
    plt.xlabel('x position (ft)')
    plt.ylabel('y position (ft)')
    
    #declare a running variable that indicates whether or not the vacuum is out of battery and to continue simulation. Updated by roomba.move()
    running = True
    
    #simulate until the roomba is out of battery
    while running:
        
        #update the position of the roomba point on the animation window
        point.set_data(roomba.xpos, roomba.ypos)
        
        #move the roomba by calling move() (see above)
        running = roomba.move()
        
        #run @ 30 fps regardless of simulation length (time is scaled by timestep length not varying framerate)
        plt.pause(1.0/30)
       
