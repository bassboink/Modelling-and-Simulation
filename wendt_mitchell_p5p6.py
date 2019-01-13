# -*- coding: utf-8 -*-
"""
wendt_mitchell_p5p6.py
Created on Sun Nov 1 14:07:05 2018
Created by: Mitchell Wendt
Revised: 11/1/2018

People who helped me: Bryan Hobocienski (For useful bits in the discussion posts, suggesting the use of np.shuffle to shuffle the array in p5)
"""

#import statements: import numpy and matplotlib to utilize arrays, random numbers, and plotting functionality
import numpy as np
import matplotlib.pyplot as plt

def p5(numTrials = 10000, numCandidates = 100):
    
    '''
    This function simulates many trials of interviewing candidates with the
    "look, then leap" hiring strategy as described in the p5 worksheet. In 
    other words, M candidates are interviewed without being hired, and then
    the next candidate who is better than the first M candidates is hired for 
    the job. 
    
    The function takes in two parameters, numTrials which is the total number
    of simulated rounds of interviews for each M value and numCandidates, the
    total number of candidates in the candidate pool. The function returns 
    nothing, but prints the M value which brings about the optimal success
    rate, based solely on the number of times when the single best candidate
    is hired.
    '''
    
    #preallocate arrays for the M values (1 through numCandidates) and the successes array that stores the number of successful trials where the top candidate is hired.
    m_vals = np.linspace(1, numCandidates, numCandidates, dtype=int)
    successes = np.zeros(numCandidates)
    
    #for every M value, run numTrials trials
    for i in m_vals:
        for j in range(0, numTrials):
            
            #initialize the array in the (randomized) order of which candidates are interviewed
            candidates = np.linspace(1, numCandidates, numCandidates, dtype=int)
            np.random.shuffle(candidates)
            
            #temporary value to store the rank of the current best interviewed candidate
            best_candidate = 0
            
            #iterate (interview) all candidates
            for k in range(0,numCandidates):
                
                #only make considerations if the candidate that just interviewed is the best candidate so far, and set the current candidate to be the best candidate
                if candidates[k] > best_candidate:
                    best_candidate = candidates[k]
                    
                    #if more than M candidates have been interviewed
                    if k > (i-1):
                        
                        #if the best candidate was selected, increment the number of successes for the given M value
                        if best_candidate == numCandidates:
                            successes[i-1] = successes[i-1] + 1
                        
                        #stop interviewing candidates and proceed to the next trial
                        break
    
    #calculate the success rate at each M value by dividing successful trials by number of trials, then plot the M value with the highest success rate
    percent_success = successes/numTrials
    print('M value with maximum success rate: ' + str(np.argmax(percent_success)+1))
    
    #plot the success rates of every M value
    plt.plot(m_vals, percent_success, 'b-')
    plt.xlabel('M value')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rates using Different M Values')

def p6(numTrials = 100000, numThrows = 100, plot_probs = False):
    
    '''
    This function simulates many trials of Kelsey Mitchell throwing free throws
    under the model where the probability of making the next throw is the 
    number of made throws over the number of attempted throws.
    
    The function takes three parameters: the number of sets of free throws
    simulated (numTrials), number of free throws in each trial (numThrows), and
    plot_probs, a boolean that determines whether or not to plot a profile of 
    probability of the 100th throw being made based on knowledge that the
    nth throw was made (set to false by default)
    
    Through setting plot_probs to be True and looking at other trials, I 
    was able to determine that the trend of the probability that Kelsey makes
    the 100th throw is only based on the number of shots where the outcome is 
    known and the number of shots where the shot is known to be made. For
    example, in part b 2 shots are known to be made (the 1st and 99th) and
    information about 3 shots (the 1st, 2nd, and 99th) is known, resulting
    in a probability of 2/3. the same goes for part c, where 6 shots are known 
    to be made (the 1st, 53rd, 54th, 55th, 56th, and 99th) and 8 shots have 
    known outcomes (1st, 2nd, 53rd, 54th, 55th, 56th, 57th, and 99th).
    '''
    
    #initialize counters for the total number of trials for each scenario (parts a, b, and c) as well as shots made in each of the scenarios
    numInA = numTrials
    numInB = 0
    numInC = 0
    numMadeInA = 0
    numMadeInB = 0
    numMadeInC = 0
    
    #if plotting the probability profile, initialize two arrays to store total number of trials and successful trials for every n where the nth throw is known to be made
    if plot_probs:
        nums = np.zeros(numThrows - 5, dtype=int)
        nums_made = np.zeros(numThrows - 5, dtype=int)
    
    #run numTrials simulations of Kelsey throwing the free throws
    for i in range(0,numTrials):
        
        #initialize an array that stores the success/fail status of every free throw and two counters that store the number of made and missed free throws
        shots = np.zeros(numThrows, dtype=int)
        made = 1
        missed = 1
        
        #after the first 2 free throws (which are known to be made and then missed, respectively), determine based on probability whether the next free throw is made using the probability made shots/total shots
        for j in range(2, numThrows):
            prob = made/(made+missed)
            rand_val = np.random.rand()
            
            #update counter and shots array if the shot is made. Note that 1-probability is used, 
            if rand_val > (1-prob):
                shots[j] = 1
                made += 1
            
            #update the missed counter if the shot is missed
            else:
                missed += 1
        
        #if the 100th throw is made, increment number of successes made for part a
        if shots[numThrows-1] == 1:
            numMadeInA += 1
            
            #if the 99th throw is additionally made, increment the number of successes made for part b
            if shots[numThrows - 2] == 1:
                numMadeInB += 1
                
                #if the criteria for part c are additionally met, increment the number of successes made for part c
                if shots[52] == 1 and shots[53] == 1 and shots[54] == 1 and shots[55] == 1 and shots[56] == 0:
                    numMadeInC += 1
            
            #if plotting the probability profile given the nth throw is made, increment the counter for each n where the nth and 100th shots are made
            if plot_probs:
                for k in range(3, numThrows-2):
                    if shots[k-1] == 1:
                        nums_made[k-3] = nums_made[k-3] + 1
        
        #if the 99th throw is made, increment the number of successes made for part b
        if shots[numThrows - 2] == 1:
            numInB += 1
            
            #if the criteria for part c are met, increment the number of successes made for part c
            if shots[52] == 1 and shots[53] == 1 and shots[54] == 1 and shots[55] == 1 and shots[56] == 0:
                numInC += 1
        
        #if plotting the probability profile given the nth throw is made, increment the counter for each n where the nth shot is made
        if plot_probs:
            for k in range(3, numThrows-2):
                if shots[k-1] == 1:
                    nums[k-3] = nums[k-3] + 1
    
    #calculate the simulated probability that the 100th shot is made for parts a, b and c
    probA = numMadeInA/numInA
    probB = numMadeInB/numInB
    probC = numMadeInC/numInC
    
    #print the probability that the 100th shot is made for parts a, b and c
    print('Probability given no other information: ' + str(probA*100) + '%')
    print('Probability given that the ' + str(numThrows-1) + 'th shot was made: ' + str(probB*100) + '%')
    print('Probability given that the 53rd, 54th, 55th, and 56th shots were made, that the 57th shot was missed, and the ' + str(numThrows-1) + 'th shot was made: ' + str(probC*100) + '%')
    
    #if printing the probability profile, calculate probabilities at each nth throw that the 100th throw is made by setting up the array of nth throws and finding the probabilities by dividing the arrays of the two counters for the nth throw
    if plot_probs:
        shot_nums = np.linspace(3,numThrows-3,numThrows-5, dtype=int)
        probs = np.zeros(numThrows - 5, dtype=float)
        for i in range(0, len(probs)):
            probs[i] = float(nums_made[i])/nums[i]
        
        #plot the results of the probabilities at each nth throw
        plt.plot(shot_nums, probs)
        plt.title('Probablility of 100th shot given the nth shot was made')
        plt.xlabel('nth shot made')
        plt.ylabel('probability')
        
                