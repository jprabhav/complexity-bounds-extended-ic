'''
This script numerically computes an individual term I(g;f(x,y)|y=j,F_j) in the extended IC statement 
for a given value of 'n' and 'j'.

The guessing probability is specified as P(g=k|f(x,y)=l,x=i,y=j):=(1+(-1)^(k+l)\epsilon) 
and thus is independent of x,y and the function's value.

'''

import numpy as np
from math import log2
import time


# Parameters

n = 4          # size of Alice's and Bob's input strings
nx = 2**n      # number of possible inputs for Alice
ny = 2**n      # number of possible inputs for Bob
delta = 1e-9   # regulator variable so that logarithms and denominators behave properly.
eps = 0.7      # bias associated with the winning probability, ranges from -1 to 1.

# Helper Functions

def base_repr(n, base, width):
    '''
    takes a number 'n' and returns its form in given a 'base' padded with zeros to the specified 'width'   
    '''
    
    if n == 0:
        return [0]*width
    digits = []
    while n:
        digits.append(int(n % base))
        n //= base
    while(len(digits) < width):
        digits.append(0)
    return digits[::-1]

def kron(i,j):
    '''
    Kronecker delta function which outputs '1' if the inputs coincide and '0' otherwise
    '''
    return (i==j)*1

# The function definition can be specified in decimal or binary form, whichever is more convenient.

def f(x, y):
    
    xbit = base_repr(x, 2, n)
    ybit = base_repr(y, 2, n)
    
    # We give a few examples of the most commonly used functios below, please uncomment them to use them.
    
    return np.dot(xbit, ybit) % 2                       # INNER PRODUCT
    #return (x == y)*1                                  # EQUALITY
    #return (x >= y)*1                                  # GREATER-THAN
    #return (np.dot(xbit,ybit) >= k)*1                  # k-INTERSECT

# Pre-compute all f values 

f_matrix = np.zeros((nx, ny), dtype=np.int8)
for i in range(nx):
    for j in range(ny):
        f_matrix[i, j] = f(i, j)
        
# Pre-compute P(g|f) matrix
pg_f = np.empty((2, 2))
pg_f[0, 0] = pg_f[1, 1] = (1 + eps) / 2
pg_f[0, 1] = pg_f[1, 0] = (1 - eps) / 2


# These distributions are only relevant when j = 0, since the register F_j is empty for j = 0.

def prob_gf_y(k, l):
    return sum(pg_f[k, l]*kron(f(i,0),l) for i in range(nx))/nx

pgf_y = np.empty((2, 2))

for k in range(2):
    for l in range(2):
        pgf_y[k,l] = prob_gf_y(k, l)

def prob_g_y(k):        
    return sum(pg_f[k, f(i,0)] for i in range(nx))/nx


# Main function 

def ic_term(j):
    """
    Here we compute the individual mutual information term I(g;f(x,y)|y=j,F_j) by breaking it into
    2 parts as I(g;f(x,y)|y=j,F_j) = H(g|y=j,F_j) - H(g|y=j,F_j,f(x,y))  
    """
    
    if j == 0:
        
        hg_y = -(prob_g_y(0) * log2(prob_g_y(0) + delta) + prob_g_y(1) * log2(prob_g_y(1) + delta))
        hg_fy = -np.sum(pgf_y * np.log2(pg_f + delta))
        
        return hg_y-hg_fy
    
    n_combinations = 2**j

    # Pre-compute entropy H(g|f=l)
    H_g_given_f = -np.sum(pg_f * np.log2(pg_f + delta), axis=0)
    
    # Generate all value combinations at once
    all_values = np.array([[(idx >> i) & 1 for i in range(j)] 
                           for idx in range(n_combinations)], dtype=np.int8)
    
    # Get relevant f_matrix columns
    f_cols = f_matrix[:, :j+1]
    
    total_ic = 0.0
    
    # Process in chunks to avoid memory explosion
    chunk_size = min(100, n_combinations)
    
    for chunk_start in range(0, n_combinations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_combinations)
        chunk_values = all_values[chunk_start:chunk_end]
        
        # Vectorized matching for entire chunk
        # Shape: (chunk_size, nx)
        matches = np.ones((chunk_end - chunk_start, nx), dtype=bool)
        
        for k in range(j):
            # Broadcasting: compare each value combination with all f values
            matches &= (f_cols[:, k] == chunk_values[:, k:k+1])
        
        # Process each combination in chunk
        for i, match_row in enumerate(matches):
            count = np.sum(match_row)
            if count == 0:
                continue
            
            p_Fj_y = count / nx
            denominator = nx * p_Fj_y + delta
            
            # Get f(i,j) values for matching rows
            f_j_values = f_cols[match_row, j]
            
            # Vectorized probability computation
            p_g_Fjy = np.array([np.sum(pg_f[k, f_j_values]) 
                                for k in range(2)]) / denominator
            
            H_g_Fjy = -np.sum(p_g_Fjy * np.log2(p_g_Fjy + delta))
            
            # P(f=l|Fj,y)
            unique, counts = np.unique(f_j_values, return_counts=True)
            p_f_Fjy = np.zeros(2)
            for val, cnt in zip(unique, counts):
                p_f_Fjy[val] = cnt / denominator
            
            # Final computation
            weighted_H = np.dot(p_f_Fjy, H_g_given_f)
            total_ic += p_Fj_y * (H_g_Fjy - weighted_H)
    
    return total_ic




start = time.time()

ic_sum = 0
for j in range(2**n):  
    ic_val = ic_term(j)
    ic_sum += ic_val
    print(f'IC({j}) = {ic_val:.7f}')
 
print('Total= ','%.7f' % ic_sum)

end = time.time()
print('The code executed in', '%.4f' % (end - start)+'s.') 