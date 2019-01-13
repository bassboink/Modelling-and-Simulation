# -*- coding: utf-8 -*-
"""
wendt_mitchell_p4.py
Created on Sun Oct 15 19:23:05 2018
Created by: Mitchell Wendt
Revised: 10/15/2018

People who helped me:
Edward Hughes (base code as provided in lecture, help with setting up the differential equations and plotting the realistic pendulum)
"""

import numpy as np                      #import numpy to calculate sin, cos, matrices, and other general mathematical functions
import matplotlib.pyplot as plt         #import matplotlib to plot the calculated pendulum positions in an animation
import scipy.integrate as integrate     #import scipy.integrate to include the differential equation solver odeint

def pendulum(thetaZero = 30, damp = 0, timeSpan = 20, length = 0.45, gravity = 9.8):
    
    '''
    This function calculates the realistic pendulum differential equation (Eq 1
    in the homework document) by splitting it into two simpler differential 
    equations (as in Eq 3 in the homework doc). The differential equations are
    solved, including the Jacobian in the solution, using the odeint solver.
    For comparison, the simple 'choir boy' solution which uses the 
    approximation that sin(theta) ~= theta (at small theta) is also included.
    
    thetaZero is the initial position of both pendulum in degrees. This is 
    converted to radians later. 30 degrees is chosen by default
    
    damp is the damping coefficient, mu, which describes how quickly a 
    pendulum's velocity slows down over time. No damping (0) is selected 
    by default.
    
    timeSpan is the length that the simulation is ran, in seconds. The 
    simulation is run for about 20 seconds by default (Note: there is a small
    amount of error in this value).
    
    length is the length of the pendulum in meters. 0.45m is used by default
    
    gravity is the coefficient of gravity in meters per second squared. By 
    default, the earth's gravity (9.8) is used
    '''
    
    if (damp < 0 or timeSpan < 0 or length < 0 or gravity < 0):   #conditions to handle value errors: if a negative damping coefficient, timeSpan, length, or gravity are input
        raise ValueError("'damp', 'timeSpan', 'length', and 'gravity' must all be greater than zero.")  #raise a value error that describes the issue to the user
        
    thetaZero = np.deg2rad(thetaZero)                             #convert initial angle theta zero from degrees to radians

    def dadt(a,t):
        
        '''
        This function defines the two equations in Eq 3 that make up the 
        equation in Eq 1. Namely, 
        
        dtheta/dt = a
        da/dt = -mu*a - g*sin(theta)/L
        
        Note that a is simply dtheta/dt, which is set to zero at the start of 
        the simulation.
        
        The function receives two inputs: a which contains a list that stores 
        dtheta/dt and theta respectively and then time t
        
        The function simply returns the two differential equations to be solved
        by odeint in a two element list.
        '''
        
        da = [a[1], -damp*a[1]-(gravity*np.sin(a[0])/length)]     #Store the two equations in a list da
        return da                                                 #return that list
    
    def jacobian(a,t):
        
        '''
        This function defines the Jacobian of the two equations in Eq 3 that 
        make up the equation in Eq 1. Namely, 
        
        dtheta/dt = a
        da/dt = -mu*a - g*sin(theta)/L
        
        The Jacobian is simply defined as the matrix of partial derivatives of
        each equation with respect to each parameter of the equations (a and t)
        
        The function receives two inputs: a which contains a list that stores 
        dtheta/dt and theta respectively and then time t
        
        The function simply returns the Jacobian of the two equation system 
        '''
        
        a1,a2 = a                                                 #retrieves the two parameters of a: dtheta/dt and theta respectively.
        dada1 = [0, 1]                                            #stores the partial derivatives of teh first equation (dtheta/dt = a) with respect to t and then a
        dada2 = [((-gravity/length)*np.cos(a1)), -damp]           #stores the partial derivatives of the second equation (da/dt = -mu*a - g*sin(theta)/L) with respect to t and then a
        jacobian = np.matrix([dada1, dada2])                      #store the lists of partial derivatives into a jacobian matrix
        return jacobian                                           #return the calculated jacobian matrix
    
    theta0 = [thetaZero, 0]                                       #defines the initial conditions of the system of differential equations: theta begins at thetaZero and dtheta/dt begins at zero
    t = np.linspace(0, timeSpan, timeSpan*30)                     #calculates the values of time t at each timestep. Evenly spaced using linspace. The 30 in this equation defines the framerate in fps that the simulation runs at
    
    theta = integrate.odeint(dadt, theta0, t, Dfun=jacobian)      #calls the odeint differential equation solver to solve the two equations in Eq 3, calls dadt as the two equations to be solved and jacobian as the jacobian matrix to be used
    x = length*np.sin(theta[:,0])                                 #calculates the x coordinate at each timestep based on its position theta. This is determined through classic "SOH CAH TOA" analysis of the coordinates and length being the hypotenuse. Theta returns 2 things: the list of theta values and a dictionary that stores other info about the solution process. Only the values are desired and extracted here.
    y = -length*np.cos(theta[:,0])                                #calculates the y coordinate at each timestep based on its position theta. Since the pivot of the pendulum is located at (0,0), a negative sign is used.
    
    choir_boy_theta = thetaZero*np.cos(np.sqrt(gravity/length)*t) #calculates theta values at each timestep using the simplified solution of the pendulum given by Eq 2 (which assumes that sin(theta) = theta)
    choir_boy_x = length*np.sin(choir_boy_theta)                  #calculates the x coordinate at each timestep as before in the real solution
    choir_boy_y = -length*np.cos(choir_boy_theta)                 #calculates the y coordinate at each timestep as before
    
    ax = plt.axes(xlim = (-1.25*length, 1.25*length),             #defines the size of the plotting window to be used by matplotlib. Note that this variable is used in pyplot but is not called explicitly in the program, which gives a warning in Spyder that the variable isn't used, even though it is used by pyplot
                  ylim = (-1.25*length, 0.25*length))
    pivot, = plt.plot(0,0)                                        #plots the pivot point of the pendulum at the origin of the plotting window. The trailing comma operator unpacks the tuple to be used by pyplot
    plt.hold(True)                                                #holds axes to allow continuous plotting on the same set of axes so that both the pendulums can be compared on the same plotting window
    
    point, = plt.plot([],[], 'r-', marker='o', label='real')                          #defines the list in which the x and y points of the realistic simulation at each timestep will be put in. ALso defines line and marker style, and label for the legend
    choirpoint, = plt.plot([], [], 'b--', marker='o', label='choir boy')              #defines the list in which the x and y points of the choir boy simulation at each timestep will be put in. ALso defines line and marker style, and label for the legend
    
    plt.legend()                                                                      #displays a legend to differentiate the realistic pendulum simulation from the choir boy simulation
    
    for xpoint,ypoint,choirxpoint,choirypoint in zip(x, y, choir_boy_x, choir_boy_y): #iterate through each x and y point of the realistic and choir boy simulations. This also effectively iterates through timesteps
        point.set_data([0, xpoint], [0, ypoint])                                      #update the x and y coordinate of the realistic solution at the given timestep
        choirpoint.set_data([0, choirxpoint], [0, choirypoint])                       #update the x and y coordinate of the choir boy solution at the given timestep
        plt.pause(0.034)                                                              #pauses the plotting for 1/30 frames per second = 0.034s, the amount of time in between frames of a 30 fps simulation
    
    plt.hold(False)                                                                   #stop hold axes to allow plotting in other figure windows outside of each instance of running this program
    
    return                                                                            #finish running the function pendulum() without returning anything
    