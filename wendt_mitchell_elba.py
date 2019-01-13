# -*- coding: utf-8 -*-
"""
wendt_mitchell_elba.py
Created on Sun Oct 23 19:23:05 2018
Created by: Mitchell Wendt
Revised: 10/25/2018

People who helped me:
Ryan Arnold (for discussing how to deal with "free riders" as a separate person, expanding the Jacobian as such, helping debug index and ode issues)
"""

#import statements
import numpy as np                      #import numpy to calculate sin, cos, matrices, and other general mathematical functions
import matplotlib.pyplot as plt         #import matplotlib to plot the deterministic and stochastic results
import scipy.integrate as integrate     #import scipy.integrate to include the differential equation solver odeint

def elba(n0 = (31999,0,1,0,0,0), nVadd = 0, tAdd = None, timeSpan = 120.0, nMax = 2000000, nRun = 1):
    
    '''
    This function calls both ode_model and Gillespie_model to simulate the spread
    of the H3N2 virus throughout the island of elba and then plot the results
    of both simulations. If one simulation of the Gillespie model is run, the
    function plots a time profile as in the deterministic solution. Otherwise,
    the second plot will be a histogram of the death toll of each individual 
    simulation.
    
    This function takes 6 parameters:
        
        n0: an integer tuple of length 6 that stores the initial values of 
        healthy people, free riders, sick people, immune people, dead people, 
        and number of vaccines respectively. 
        
        nVadd: the number of vaccines added to the simulation at time tAdd. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        tAdd: the time (in days) at which additional vaccines are supplied. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        timeSpan: the maximum time (in days) that the simulations can run for
        
        nMax: the maximum number of reactions in the Gillespie simulation, also
        the number of timesteps taken in the ode simulation
        
        nRun: the number of stochastic simulations being run. If 1, a time 
        profile of people will be plotted as in the deterministic solution.
        Otherwise, a histogram of the death toll of each simulation is plotted
        instead
        
    This function returns nothing.
    '''
    
    #Error Handling
    
    #throw an error if a tuple passed to n0 is not of length 6
    if len(n0) != 6:
        raise ValueError("Exactly 6 inputs must be entered in n0: healthy people willing to be vaccinated, free riders, sick people, immune people, dead people and doses of vaccine at the start of the simulation")
    #throw an error if any of the values inside n0 are negative
    for i in range (0, len(n0)):
        if n0[i] < 0:
            raise ValueError("Inputs to n0 cannot be negative")
    #throw an error if a negative timeSpan or nMax are passed
    if (nVadd < 0 or timeSpan < 0 or nMax < 0):
        raise ValueError("nVadd, timeSpan and nMax cannot have negative values")
    #throw an error if a negative value or zero is passed to nRun
    if nRun < 1:
        raise ValueError("nRun must be 1 or greater")
    #throw an error if tAdd is not in between 0 and timeSpan if vaccines are added
    if nVadd > 0:
        if (tAdd == None or tAdd <= 0 or tAdd >= timeSpan):
            raise ValueError("If additional vaccines are being added, tAdd must be greater than zero and less than the timeSpan")
    
    #Calculations
    
    #call ode_model to fetch t and n for the deterministic solution
    t_deterministic, n_deterministic = ode_model(n0, nVadd, tAdd, timeSpan, nMax)
    
    #initialize death_tolls if more than one stochastic simulation is running
    if nRun > 1:
        death_tolls = np.array([], dtype=int)
    
    #call Gillespie_model nRun times to fetch t and n data for the stochastic solution
    for i in range (0, nRun):
        t_gillespie, n_gillespie = Gillespie_model(n0, nVadd, tAdd, timeSpan, nMax)
        #retrieve the death toll of each solution and store in death_tolls if more than 1 stochastic solution is running
        if nRun > 1:
            death_tolls = np.append(death_tolls, n_gillespie[len(t_gillespie)-1][4])
    
    #Plotting
    
    #set up the plotting window to fill the screen and have two subplots for the deterministic and stochastic solutions
    plt.figure(figsize=(15,8))
    plt.subplot(1, 2, 1)
    
    #plot the number of healthy people, sick people, immune people, and dead people over time on the plot on the left
    plt.plot(t_deterministic, n_deterministic[:,0] + n_deterministic[:,1], 'b-', label = 'healthy')
    plt.plot(t_deterministic, n_deterministic[:,2], 'r-', label = 'sick')
    plt.plot(t_deterministic, n_deterministic[:,3], 'g-', label = 'immune')
    plt.plot(t_deterministic, n_deterministic[:,4], 'k-', label = 'dead')
    
    #set titles, axes labels, and legend of first plot
    plt.title('Deterministic Solution: Time Profiles')
    plt.xlabel('t (days)')
    plt.ylabel('Number of people')
    plt.legend()
    
    #if only one stochastic simulation is running, plot a time profile as in the deterministic solution on the right
    if nRun == 1:
        plt.subplot(1, 2, 2)
        
        #plot the number of healthy people, sick people, immune people, and dead people over time
        plt.plot(t_gillespie, n_gillespie[:,0] + n_gillespie[:,1], 'b-', label = 'healthy') #note here that both healthy people and free riders are included here
        plt.plot(t_gillespie, n_gillespie[:,2], 'r-', label = 'sick')
        plt.plot(t_gillespie, n_gillespie[:,3], 'g-', label = 'immune')
        plt.plot(t_gillespie, n_gillespie[:,4], 'k-', label = 'dead')
        
        #set titles, axes labels, and legend of second plot
        plt.title('Gillespie Solution: Time Profiles')
        plt.xlabel('t (days)')
        plt.ylabel('Number of people')
        plt.legend()
    
    #if more than one stochastic simulation is running, plot a histogram of the death tolls on the right instead of the time profile
    else:
        plt.subplot(1, 2, 2)
        plt.hist(death_tolls, bins='auto')
        
        #set titles and axes labels of second plot
        plt.title('Gillespie Solution: Death Tolls at End of Simulation')
        plt.xlabel('Number of Dead People')
        plt.ylabel('Frequency of Death Toll Range')
    
    #end program without returning anything
    return

def ode_model(n0, nVadd, tAdd, timeSpan, nMax):
    
    '''
    This function solves the system of ODEs to predict the number of healthy,
    sick, immune, and dead people over time. It solves this system:
        
        dhealthy/dt = -k1*healthy*sick - k4*healthy*vaccines
        dfree_riders/dt = -k1*free_riders*sick
        dsick/dt = k1*healthy*sick + k1*free_riders*sick - k2*sick - k3*sick
        dimmune/dt = k2*sick + k4*healthy*vaccines
        ddead/dt = k3*sick
        dvaccines/dt = -k4*healthy*vaccines
        
    The function solves the differential equations with the initial conditions
    n0, and uses the jacobian of the differential equations to solve the system
    as well. If vaccines are bieng added, the set of equations needs to be 
    solved in two calls to odeint: one before and one after the vaccines are 
    added.
    
    This function takes 5 parameters:
        
        n0: an integer tuple of length 6 that stores the initial values of 
        healthy people, free riders, sick people, immune people, dead people, 
        and number of vaccines respectively. 
        
        nVadd: the number of vaccines added to the simulation at time tAdd. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        tAdd: the time (in days) at which additional vaccines are supplied. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        timeSpan: the maximum time (in days) that the simulation can run for
        
        nMax: the maximum number of reactions in the Gillespie simulation, also
        the number of timesteps taken in the ode simulation
        
    This function returns an array t (which stores the values of time at each
    timestep) and n (which stores the number of healthy people, free riders, 
    sick people, dead people, immune people, dead people, and vaccines at 
    each timestep).
    '''
    
    #initialize the k values for the H3N2 virus
    k1 = 0.0000176
    k2 = 0.1
    k3 = 0.01
    k4 = 0.00000352
    
    #initialize the arrays for the t and n values
    t = np.linspace(0, timeSpan, nMax)
    n = np.zeros((nMax, 6))
    
    #initialize t_first and n_first if vaccines are being added to store the t and n values before the vaccines are added
    if nVadd > 0:
        t_first = np.linspace(0, tAdd, int(nMax*tAdd/timeSpan))
        n_first = np.zeros((int(nMax*tAdd/timeSpan), 6))
    
    def dydx(n_vals, t):
        
        '''
        This function returns the system of equations to be solved by odeint:
        
        dhealthy/dt = -k1*healthy*sick - k4*healthy*vaccines
        dfree_riders/dt = -k1*free_riders*sick
        dsick/dt = k1*healthy*sick + k1*free_riders*sick - k2*sick - k3*sick
        dimmune/dt = k2*sick + k4*healthy*vaccines
        ddead/dt = k3*sick
        dvaccines/dt = -k4*healthy*vaccines
        
        This function takes 2 inputs, n_vals and t for the t and n data and
        returns the list of differential equations to be solved, dydt
        '''
        
        #split the n vector into healthy people, free riders, sick, immune, dead people, and vaccines
        healthy, free_riders, sick, immune, dead, vaccines = n_vals
        
        #list each equation, given in Worksheet 3
        eq1 = -k1*healthy*sick - k4*healthy*vaccines
        eq2 = -k1*free_riders*sick #note that eq2 is for free riders, a healthy person that refuses to get vaccinated. In order the indexes of n represent healthy people, free riders, sick people, immune people, dead people, and vaccines respectively. This convention follows in all other instances of n as well.
        eq3 = k1*(healthy + free_riders)*sick - k2*sick - k3*sick
        eq4 = k2*sick + k4*healthy*vaccines
        eq5 = k3*sick
        eq6 = -k4*healthy*vaccines
        
        #combine the equations into a single list and return it
        dydt = [eq1, eq2, eq3, eq4, eq5, eq6]
        return dydt

    def Jacobian(n_vals, t):
        
        '''
        This function returns the jacobian of the system of equations above,
        found by calculating partial derivatives with respect to each variable.
        The initial Jacobian is provided in Worksheet 3, but additional elements
        were required since the free riders are treated as a separate type of
        person.
        
        This function takes 2 inputs, n_vals and t for the t and n data and 
        returns the jacobian matrix of the system of equations above, found
        through partial derivatives.
        '''
        
        #split the n vector into healthy people, free riders, sick, immune, dead people, and vaccines
        healthy, free_riders, sick, immune, dead, vaccines = n_vals
        
        #list the partial derivatives of each equation with respect ot each variable
        row1 = [-k1*sick-k4*vaccines, -k1*sick-k4*vaccines, -k1*healthy-k1*free_riders, 0, 0, -k4*healthy]
        row2 = [-k1*sick-k4*vaccines, -k1*sick-k4*vaccines, -k1*healthy-k1*free_riders, 0, 0, 0]
        row3 = [k1*sick, k1*sick, k1*healthy+k1*free_riders-k2-k3, 0, 0, 0]
        row4 = [k4*vaccines, 0, k2, 0, 0, k4*healthy]
        row5 = [0, 0, k3, 0, 0, 0]
        row6 = [-k4*vaccines, 0, 0, 0, 0, -k4*healthy]
        
        #combine the lists into a matrix and return the matrix
        jacobian = np.matrix([row1, row2, row3, row4, row5, row6])
        return jacobian
    
    #if vaccines are being added, the system of equations needs to be solved twice
    if nVadd > 0:
        
        #solve the system up to tAdd
        n_first = integrate.odeint(dydx, n0, t_first, Dfun=Jacobian)
        
        #add the vaccines
        n_first[int(nMax*tAdd/timeSpan)-1][5] = n_first[int(nMax*tAdd/timeSpan)-1][5] + nVadd
        
        #create n_after (solving the system of equations again) and t_after arrays, to be concatenated with n_first and t_first respectively
        t_after = np.linspace(tAdd, timeSpan, int(nMax*(timeSpan-tAdd)/timeSpan))
        n_after = integrate.odeint(dydx, n_first[int(nMax*tAdd/timeSpan)-1,:], t_after, Dfun=Jacobian)
        
        #concatenate the arrays to get the total n and t data to be returned and plotted
        n = np.concatenate((n_first, n_after))
        t = np.linspace(0, timeSpan, len(n_first)+len(n_after))
        
    #if vaccines are not being added, all than needs to be done is to solve the system of equations once, storing data in n and t
    else:
        n = integrate.odeint(dydx, n0, t, Dfun=Jacobian)
    
    #return the t and n data to the elba() function
    return t,n

def Gillespie_model(n0, nVadd, tAdd, timeSpan, nMax):
    
    '''
    This function simulates the spread of H3N2 using the Gillespie algorithm
    to stochastically determine the number of healthy, free rider, sick, 
    immune, and dead people over time. 
    
    This function takes 5 parameters:
        
        n0: an integer tuple of length 6 that stores the initial values of 
        healthy people, free riders, sick people, immune people, dead people, 
        and number of vaccines respectively. 
        
        nVadd: the number of vaccines added to the simulation at time tAdd. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        tAdd: the time (in days) at which additional vaccines are supplied. If
        no vaccines supplied, tAdd should be None and nVadd should be 0
        
        timeSpan: the maximum time (in days) that the simulation can run for
        
        nMax: the maximum number of reactions in the Gillespie simulation, also
        the number of timesteps taken in the ode simulation
        
    This function returns an array t_vals (which stores the values of t at each
    timestep) and n (which stores the number of healthy people, free riders, 
    sick people, dead people, immune people, dead people, and vaccines at 
    each timestep).
    '''
    
    #define the rate constants of the H3N2 virus
    k1 = 0.0000176
    k2 = 0.1
    k3 = 0.01
    k4 = 0.00000352
    
    #preallocate arrays for the t and n values
    t_vals = np.zeros(nMax)
    n = np.zeros((nMax, 6))
    
    #set the first line of n to be n0, setting the initial condition
    n[0,:] = n0
    
    #declare t and index as the current time and number of reactions respectively
    t = 0
    index = 1
    
    #declare a boolean added to determine if additional vaccines have been added yet or not. Only applies when tAdd is not equal to None
    added = False
    
    #loop until the maximum number of reactions are run (or if a break statement is encountered)
    while index <= nMax:
        
        #calculate the rates of each 'reaction' as given in Worksheet 3
        r1 = k1*n[index-1][0]*n[index-1][2]
        r2 = k1*n[index-1][1]*n[index-1][2]
        r3 = k2*n[index-1][2]
        r4 = k3*n[index-1][2]
        r5 = k4*n[index-1][0]*n[index-1][5]
        
        #calculate rtotal as the sum of all rates and finish the simulation if rtotal = 0 (no reactions can happen anymore, this happens when everyone is either immune or dead)
        rtotal = r1 + r2 + r3 + r4 + r5
        if rtotal == 0:
            
            #truncate the length of the n and t arrays, as nMax reactions were not achieved
            n = np.delete(n, np.s_[(index):], 0)
            t_vals = np.delete(t_vals, np.s_[(index):], 0)
            
            #break from the loop, end simulation
            break
        
        #generate a random number from 0 to 1 to determine the interval of time at this timestep, as per the Gillespie algorithm
        rand_num = np.random.rand()
        
        #use the formula given in the Gillespie slides to determine the length of time of the current timestep, increment t by this value
        tau = -np.log(rand_num)/rtotal
        t = t + tau
        
        #break from the loop if the maximum timeSpan is reached for the simulation
        if t > timeSpan:
            
            #truncate the n and t_vals arrays as nMax reactions were not achieved
            n = np.delete(n, np.s_[(index):], 0)
            t_vals = np.delete(t_vals, np.s_[(index):], 0)
            
            #break from the loop, end simulation
            break
        
        #store the current t value in the preallocated array
        t_vals[index] = t
        
        #if vaccines are to be added and haven't been added yet, check if the current time is past tAdd and then add the vaccines if this is the case
        if tAdd != None and not added:
            if t > tAdd:
                n[index][5] = n[index][5] + nVadd
                added = True
        
        #based on the calculated rate and the sum of all rates, calculate the probability that each reaction is going to occur at the current timestep
        p = [r1/rtotal, r2/rtotal, r3/rtotal, r4/rtotal, r5/rtotal]
        
        #calculate the csp vector, the sum of probabilities (cumulative probabilities)
        csp = [0]*5
        sum_p = 0
        for i in range(0,5):
            sum_p += p[i]
            csp[i] = sum_p
        
        #generate another random value from 0 to 1 to be used to determine which reaction occurs based on the cumulative probabilities of each reaction, as per the '1D dartboard' model
        q = np.random.rand()
        
        #declare v, the change in each n value at the current timestep based on the stoichiometry of which reaction occurs, determined by q (randomly generated)
        v = []
        if q < csp[0]:
            v = [-1, 0, 1, 0, 0, 0]
        elif q < csp[1]:
            v = [0, -1, 1, 0, 0, 0]
        elif q < csp[2]:
            v = [0, 0, -1, 1, 0, 0]
        elif q < csp[3]:
            v = [0, 0, -1, 0, 1, 0]
        else:
            v = [-1, 0, 0, 1, 0, -1]
        
        #set the n values based on v and the n values at the previous timestep
        n[index] = n[index-1] + v
        
        #increment the number of reactions
        index += 1
    
    #return the t and n data to be plotted
    return t_vals,n