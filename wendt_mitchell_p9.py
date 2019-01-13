# -*- coding: utf-8 -*-
"""
wendt_mitchell_p9.py
Created on Sun Nov 5 22:56:33 2018
Created by: Mitchell Wendt
Revised: 11/9/2018

People who helped me: Bryan Hobocienski (For providing the useful bit of code to import the csv English Word data as a tuple)
                      Jim Rathman (For providing code for the seekuence function in the slides to base the evolutionary algorithm off of)
                      Ryan Arnold (For profound discussion of scoring approaches, issues regarding spaces and word rankings)
"""

import pandas as pd       #import pandas for use in importing the data from the excel file
import string             #import string to import ascii characters to be used
import random             #import random to implement random operations on lists
import numpy as np        #import numpy for the use of binomial probability distibution in determining random mutations

#set the location of the Excel English word data
csvFile = 'C:/Users/Mitchell/Downloads/English_words.csv'

#read in the Excel data using pandas as a dataframe
englishWords = pd.read_csv(csvFile)

#converting pandas dataframes to tuples
rank = tuple(englishWords.loc[:, 'Rank'])
word = tuple(englishWords.loc[:, 'Word'])
pos = tuple(englishWords.loc[:, 'Part of speech'])
freq = tuple(englishWords.loc[:, 'Frequency'])
disp = tuple(englishWords.loc[:, 'Dispersion'])

#count up the total number of word instances found in the list (used to generate the frequency factor, found later)
tot_word_freq = sum(freq)

#generate a large string that contains every usable character in random generation
char = string.ascii_letters + string.digits + string.punctuation + ' '

def evolver(parent = 'Beware of ManBearPig!', nGen = 1000, nChildren = 20, \
            mutationProbs = (0.01, 0.002, 0.001), printGens = False):
    
    '''
    This function runs the evolutionary algorithm on a starting string for a
    given number of generations. This portion is largely based off of the 
    slides provided by Jim Rathman (slide 381).
    
    The function takes a parent (string), number of generations (int), number
    of children (int), mutation probabilities (tuple of 3 ints), and 
    whether to print out output (bool)
    
    The function returns the final string at generation nGen
    '''
    
    #Error Handling
    
    #Check if the parent is a string
    if type(parent) is not str:
        raise TypeError('Query sequence must be a string')
    
    #Check if there are three mutation probabilities given
    elif len(mutationProbs) != 3:
        raise ValueError('mutationProbs must be a tuple of 3 floats')
    
    #Check that the mutation probabilities are indeed floats
    elif type(mutationProbs[0]) is not float or type(mutationProbs[1]) is not float or type(mutationProbs[2]) is not float:
        raise TypeError('mutationProbs must be a tuple of 3 floats')
    
    #Check if nGen and nChildren are integers
    elif type(nGen) is not int or type(nChildren) is not int:
        raise TypeError('nGen and nChildren must be integers')
    
    #Check if nGen, nChildren, and the given probabilities are all greater than 0
    elif nGen < 0 or nChildren < 0 or mutationProbs[0] < 0 or mutationProbs[1] < 0 or mutationProbs[2] < 0:
        raise ValueError('nGen, nChilden, and probabilities must be greater than zero')
    
    #Check that the printGens parameter is a boolean
    elif type(printGens) is not bool:
        raise TypeError('printGens must be a boolean')
    
    #Split the mutation probabilities tuple and start a generation counter
    subProb, delProb, insProb = mutationProbs
    generation = 0
    
    #If 'random' is entered, generate a string of 15 random characters
    if parent == 'random':
        parent = ''.join(random.choice(char) for x in range(15))
    
    #Loop until the generation count is reached
    while generation < nGen:
        
        #Initiate score to None
        score = None
        
        #For every child, go through the parent string and randomly mutate
        for i in range(nChildren):
            child = ''
            
            #Check through mutations at each character of the parent
            for a in parent:
                
                #Randomly choose whether a substitution, deletion, or insertion could happen
                mutation = random.choice(('sub', 'del', 'ins'))
                
                #If substitution and binomial probability randomly reached, swap the current character
                if mutation == 'sub' and np.random.binomial(1, subProb) == 1:
                    a = random.choice(char)
                
                #If deletion and binomial probability randomly reached, delete the current character
                elif mutation == 'del' and np.random.binomial(1, delProb) == 1:
                    a = ''
                
                #If insertion and binomial probability randomly reachedm, either insert a random character before or after the current character
                elif mutation == 'ins' and np.random.binomial(1, insProb) == 1:
                    side = random.choice(('before', 'after'))
                    if side == 'before':
                        a = random.choice(char) + a
                    else:
                        a = a + random.choice(char)
            
                #Update the current child
                child = child + a
            
            #Score the child string
            tem = seqscore(child)
            
            #If there hasn't been a score yet or if the child score is higher, update the score and parent
            if score is None or tem > score:
                score = tem
                next_parent = child
        
        #Update the parent after all children are generated, increment the generation counter
        parent = next_parent
        generation += 1
        
        #Output the results of the generation
        if printGens:
            print('Gen ', generation, '\tScore = ', score, '\t', parent)
    
    #Return the final string
    return parent

def seqscore(inseq = None):
    
    '''
    This function is the scoring algorithm used in determining whether certain
    strings are better than others. The function simply takes in the string and
    returns the score as a float.
    
    The function mainly uses the given dictionary to score higher, with special
    consideration given to spacing and punctuation. See comments below for 
    details
    '''
    
    #Throw an error if a string is not passed
    if type(inseq) is not str:
        raise TypeError('Query sequence must be a string!')
    
    #Initialize the score to zero
    score = 0
    
    #Variables initialized in determining 'terms', or characters in between spaces
    in_a_term = False
    start_term_index = 0
    end_term_index = 0
    
    #Iterate through the characters of the string first
    for i in range(0, len(inseq)):
        
        #Determine if a term is started or not, store index
        if inseq[i] != ' ' and not in_a_term:
            start_term_index = i
            in_a_term = True
        
        #Determine if a term is ending
        elif inseq[i] == ' ' and in_a_term:
            end_term_index = i
            in_a_term = False
            
            #Score highly if the term is a word
            if inseq[start_term_index:end_term_index] in word:
                score += 250
            
            #Score lowly if the term is one character and is not 'a' or 'I'
            elif end_term_index-start_term_index == 1 and inseq[start_term_index:end_term_index] != 'I' \
            and inseq[start_term_index:end_term_index] != 'a':
                score = score -100
        
        #Subtract points for digits and punctuation characters that are not punctuation for a sentence
        if inseq[i] in string.punctuation or inseq[i] in string.digits and inseq[-1] != '.' \
        and inseq[-1] != '!' and inseq[-1] != '?' and inseq[-1] != ','and inseq[-1] != "'":
            score = score - 25
        
        #If a comma is used, score highly if it has a space after it
        elif inseq[i] == ',' and i != len(inseq)-1:
            if inseq[i+1] == ' ':
                score += 50
        
        #Whenever spaces are present, score a little higher
        elif inseq[i] == ' ':
            score += 10
            
            #Doc points if there are trailing spaces
            if i != len(inseq)-1:
                if inseq[i+1] == ' ':
                    score = score - 50
    
    #Score highly if there is normal sentence punctuation at the end of the sentence (scores are high for this since the algorithm was resistant)
    if inseq[-1] == '.' or inseq[-1] == '!' or inseq[-1] == '?':
        score += 1000
        
        #More special consideration if there is a period
        if inseq[-1] == '.':
            score += 5000
    
    #Now iterate through each dictionary word
    for i in range(0,len(word)):
        
        #When there is a word found in the string, score it higher
        if word[i] in inseq:
            
            #Calculate the frequency factor (approximate percentage of the time the word is used in language)
            proportion_used = float(freq[i])/tot_word_freq
            
            #Set a vocabulary multiplier if the word rank is high; sophisticates sentences
            multiplier = 1
            if rank[i] > 250:
                multiplier = 10
            
            #Increment score based on frequency used, word length, and vocabulary factor
            score += 10000*proportion_used*len(word[i])*multiplier
            
            #Add to the score if there is a space after the word
            if inseq.find(word[i])+len(word[i]) < len(inseq):
                if inseq[inseq.find(word[i])+len(word[i])] == ' ':
                    score += 50
    
    #Return the final score value
    return score