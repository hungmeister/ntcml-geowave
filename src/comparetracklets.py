## Contains functions for performing chi squared independence tests 
## on a contingency table that is generated from observations of two track fragments

import sys
import numpy as np
import os
import scipy.stats as stats
import random

# test track fragments
raw1 = np.random.rand(10,4)
raw2 = np.random.rand(15,4)

# other useful parameters
rejectprob = 0.95

# build contingency table for chi2 test for independence
def chi2_buildtable(dist1, dist2): #dists are assumed to be numpy arrays, where rows represent unique observations and columns represent feature dimensions
    dist_mins = np.amin(np.concatenate((dist1,dist2), axis = 0), axis = 0)
    dist_maxs = np.amax(np.concatenate((dist1,dist2), axis = 0), axis = 0)
    # instantiate an empty contingency table with two rows corresponding to our two distributions and columns for every bin, where 'low' and 'high bins' exist for every combination of features, 
    table = np.zeros((2,2**dist1.shape[1]))
    
    # loop over all columns in the contingency table in order to encode dist1 and dist2 as a histogram 
    for j in range(0,table.shape[1]+1):
        #binaryvectorj = np.array(list(np.binary_repr(j)),dtype=int) #convert j into an array whose elements are its binary digits
        binaryvectorj = np.binary_repr(j) #convert j into a string representing its binary digits
        
        # loop over all observations for the first distribution
        for k in range(0,dist1.shape[0]+1):
            #instantiate a zero vector to store the bin assignment for a given observation and populate by looping through the dimensions
            binassignment = np.zeros((dist1.shape[1],),dtype = int)
            
            # loop over all dimensions of the observation
            for l in range(0,dist1.shape[1]+1): # assign a particular observation to a bin represented as a binary number
                if dist1.item(j,k) > (dist_mins.item(k) + dist_maxs.item(k))/2: # assign a particular feature to 
                    binassignment = 1 # otherwise, the bit remains 0
            # add 1 to the corresponding bin if the test condition is met
            #the problem with using the np.all() function is that the binary values being compared may be of different lengths, which will require padding with zeros
            # Intent is to compare binaryvectorj to binassignment, and add 1 to the appropriate element in table if they are equal
            binassignmentstring = ''.join(map(str, binassignment)) #NEED TO VERIFY THAT THIS DOESNT INTRODUCE EXTRA SPACES OR OTHER CHARACTERS PREVENTING AN EXACT MATCH
            if binvectorj == binassignmentstring: 
                table[0,j] = table[0,j] + 1
                
        # repeat the k and l loop for dist2
        
        # loop over all observations for the second distribution
        for k in range(0,dist2.shape[0]+1):
            #instantiate a zero vector to store the bin assignment for a given observation and populate by looping through the dimensions
            binassignment = np.zeros((dist2.shape[1],),dtype = int)
            
            # loop over all dimensions of the observation
            for l in range(0,dist2.shape[1]+1): # assign a particular observation to a bin represented as a binary number
                if dist2.item(j,k) > (dist_mins.item(k) + dist_maxs.item(k))/2: # assign a particular feature to 
                    binassignment = 1 # otherwise, the bit remains 0
            # add 1 to the corresponding bin if the test condition is met
            #the problem with using the np.all() function is that the binary values being compared may be of different lengths, which will require padding with zeros
            # Intent is to compare binaryvectorj to binassignment, and add 1 to the appropriate element in table if they are equal
            binassignmentstring = ''.join(map(str, binassignment)) #NEED TO VERIFY THAT THIS DOESNT INTRODUCE EXTRA SPACES OR OTHER CHARACTERS PREVENTING AN EXACT MATCH
            if binvectorj == binassignmentstring: 
                table[0,j] = table[0,j] + 1
    return table

# Compare binned observations of two track fragments using Pearson's chi squared test and Welch's t test
def chi2_run(ctgy_table, prob):
    
    # Pearson's chi squared test
    chi_stat, chi_p, dof, expected = stats.chi2_contingency(ctgy_table)
    alpha = 1.0 - prob
    if chi_p <= alpha:
        print('Chi2: Dependent (reject H0)')
    else:
        print('Chi2: Independent (fail to reject H0)')

def main():
    test_table = chi2_buildtable(raw1,raw2)
    chi2_run(test_table,rejectprob)

main()

#TO DO:
# Fix the chi2_buildtable function (it is producing an error at binassignmentstring = ''.join(map(str, binassignment)) )
# Delete columns from the contingency table that contain all zeros
# Aggregate columns that contain small counts (I don't know if there is a way to do this that isn't naive)