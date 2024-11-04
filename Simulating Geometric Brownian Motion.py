import numpy as np
import matplotlib.pyplot as plt

''' Parameters '''

mu = 0.1 # drift coefficient, representing the expected return of the asset per year

n = 10000 # number of steps

t = 1

M = 50 # number of simulation paths (aka trajectories) for the asset price

S0 = 100

sigma = 0.3


''' Simulating Geometric Brownian Motion Paths '''

dt = t / n  # length of each time step within the one-year period

# simulation using numpy arrays
St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size = (M, n)).T)
'''
Notes:

(mu - signma ** 2 / 2) * dt is the mean-adjustment term which adjusts for the drift and volatility. this adjustment centers the distribution of returns

sigma * np.random.normal(0, np.sqrt(dt), size = (M, n)) is the random term which simulates the random fluctuations of the stock price

the random term generates a matrix of normally distributed random variables with a mean of 0 and a s.d. of sqrt(dt) for each time step

transpose is required to match the times series structure

exponential function is applied to convert these normally distributed changes into multiplicative factors (since GBM assumes a log-normal process)
'''


St = np.vstack([np.ones(M), St]) # vstack is used for stacking arrays vertically with more than 1 dimension, as long as the # of columns matches
# so basically here, its stacking 1 in the first row to make all the stock prices start at 1 (later scaled by S0)

St = S0 * St.cumprod(axis = 0) # .cumprod() takes the cumulative product along each path (axis 0), simulating the compound growth of the stock price over each time step
# resulting an (n + 1) * M array where each column represents one possible stock price path over the n steps


''' Constructing time intervals '''

time = np.linspace(0, t, n+1) # generate num (in this case n + 1) equally spaced points betwee start, 0, and stop, T
# so, time is an array representing discrete time points over the simulation period (from 0 to T)

tt = np.full(shape = (M, n+1), fill_value = time).T # np.full() creates an array of a specified shape, filing it with the provided fill_value
# shape = (M, n+1) <-- M rows and n + 1 columns
# fill_value = time <-- fills every column with the time array
# since I need the matrix to be (n + 1) * M in order to match the matrix of St, transpose is needed


''' Plotting the results '''

plt.plot(tt, St)
plt.xlabel('Years $(t)$')
plt.ylabel('Stock Price $(S_t)$')
plt.title(f'Realizations of Geometric Brownian Motion\n $dS_t = \\mu S_t dt + \\sigma S_t dW_t$\n $S_0 = {S0}, \\mu = {mu}, \\sigma = {sigma}$')
plt.show()