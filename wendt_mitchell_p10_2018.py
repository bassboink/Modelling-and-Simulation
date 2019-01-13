# -*- coding: utf-8 -*-
"""
wendt_mitchell_p10_2018.py
Created on Sun Nov 15 01:10:22 2018
Created by: Mitchell Wendt
Revised: 11/20/2018

People who helped me: Ryan Arnold (For bouncing ideas off each other and playing test matchups against each other.
                                   This is not a conspiracy theory I promise! You can read the code yourself)
"""

#import numpy for random number generation
import numpy as np

def wendt_mitchell_p10_2018(history, score):
    
    '''
    This function contains my strategy for maximizing score in the iterated 
    prisoner dilemma problem. It takes a history list (showing previous input
    of the player and the opponent) and a scores tuple that scores the current
    score of each player. The function simply returns either 'h' or 'd'
    depending on the situation. 
    
    The method itself is a bit of a Monte Carlo method, but not completely.
    The first three plays are fixed, and most other plays are involved
    with the deterministic 'tit for tat' strategy where the function simply 
    returns the opponent's last play. However, it also incorporates the 
    probabilistic method, where it plays based on the calculated probability 
    that the opponent's next play will be either 'h' or 'd'. Choosing these
    methods is also itself random/stochastic in nature
    '''
    
    #potentially unnecessary error handling
    if history is not None and not isinstance(history, (list,)):
        raise TypeError("history should be a list of two character strings")
    elif score is not None and not isinstance(score, (tuple,)):
        raise TypeError("score should be a two member integer tuple")
    elif score is not None and len(score) != 2:
        raise TypeError("score should be a two member integer tuple")
    for i in range(0, len(score)):
        if type(score[i]) is not int:
            raise TypeError("score should be a two member integer tuple")
    for i in range(0, len(history)):
        if type(history[i]) is not str:
            raise TypeError("history should be a list of two character strings")
        elif len(history[i]) is not 2:
            raise ValueError("history should be a list of two character strings")
        elif history[i][0] != 'h' and history[i][0] != 'd':
            raise ValueError("history should be a list of two character strings with only 'd' and 'h' as values")
        elif history[i][1] != 'h' and history[i][1] != 'd':
            raise ValueError("history should be a list of two character strings with only 'd' and 'h' as values")
    
    #play 'h', 'd', 'd' as the first 3 plays. This is because I expect a lot of first plays to be 'd', but I also want to set up for a possible 'dd' streak
    if history is None:
        return 'h'
    elif len(history) == 1 or len(history) == 2:
        return 'd'
    
    #play up to 25 rounds using the tit for tat strategy to start out. It's a solid strategy to begin with anyway
    elif len(history) < 25:
        return history[-1][1]
    
    #do this for the next 175 games...
    else:
        
        #count the number of hawks and doves the opponenet played
        numDove = 0
        numHawk = 0
        for i in range(0,len(history)):
            if history[i][1] == 'd':
                numDove += 1
            else:
                numHawk += 1
        
        #the critical bit that differentiates my strategy from tit for tat: I call it a "crunch". Basically after about 70% of the way through games, 
        #if the player has been cooperating for a while I slam them with a bunch more hawks (80% chance). It is done randomly so that other methods like tit for 
        #tat or probabilistic methods cannot adjust. As Dr. Clay would say, "Love it!"
        if len(history) > 140 and history[-1][1] == 'd' and history[-2][1] == 'd':
            rand_num = np.random.rand()
            if rand_num > 0.2:
                return 'h'
            else:
                return 'd'
        
        #match whatever the player has been playing if they are close to either all doves or all hawks (I left a little wiggle room with the > 3)
        elif numDove < 3:
            return 'h'
        elif numHawk < 3:
            return 'd'
        
        #also match whatever the player has been playing if there is currently a streak of 'h' or 'd' (5 or more). 
        #I'm assuming that these won't be so random in the opponents I play, but even if they are it adjusts quickly at the next iteration.
        elif history[-1][1] == 'd' and history[-2][1] == 'd' and history[-3][1] == 'd' and history[-4][1] == 'd' \
        and history[-5][1] == 'd':
            return 'd'
        elif history[-1][1] == 'h' and history[-2][1] == 'h' and history[-3][1] == 'h' and history[-4][1] == 'h' \
        and history[-5][1] == 'h':
            return 'h'
        
        #default strategy: 85% tit for tat 15% probabilistic
        else:
            rand_num = np.random.rand()
            
            #85% tit for tat
            if rand_num > 0.15:
                return history[-1][1]
            
            #15% probabilistic
            else:
                randomer_num = np.random.rand()
                
                #compare the random number with the proportion of all opponent dove plays 
                if randomer_num < (float(numDove)/(numDove+numHawk)):
                    return 'd'
                else:
                    return 'h'