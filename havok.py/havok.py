"""
Package to apply havok to a single data stream
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import hankel, svd, lstsq
from sklearn.preprocessing import normalize
from control.matlab import *
#import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def opt_SVHT_coef(beta,num):
    # https://www.researchgate.net/publication/236935805_The_Optimal_Hard_Threshold_for_Singular_Values_is
    # Typically we pick a max rank below this, and it's hard to program. 4.5 is the max the function will return
    
    return 4.5 #optSVHT

def thetaGen(x,order,usesine):
    n, nVars = x.shape # num of rows, number of columns
    
    # order zero
    yout = np.ones((n,1))
    
    # order one
    yout = np.hstack((yout,x))
    
    # order two
    if order >= 2:
        for i in np.arange(0.,nVars): 
            for j in np.arange(i,nVars):
                y2 = (x[:,int(i)] * x[:,int(j)])[:, np.newaxis]
                yout = np.hstack((yout,y2)) # x[:,i]*x[:,j]
                
    if order >= 3:
        n2, y2Vars = y2.shape
        for i in np.arange(0.,nVars):
            for j in np.arange(i,y2Vars):
                y3 = (x[:,int(i)] * y2[:,int(j)])[:, np.newaxis]
                yout = np.hstack((yout,y2))
                
    if order >= 4:
        n3, y3Vars = y3.shape
        for i in np.arange(0.,nVars):
            for j in np.arange(i,y3Vars):
                y4 = (x[:,int(i)] * y3[:,int(j)])[:, np.newaxis]
                yout = np.hstack((yout,y4))

    
    if order >= 5:
        n4, y4Vars = y4.shape
        for i in np.arange(0.,nVars):
            for j in np.arange(i,y4Vars):
                y5 = (x[:,int(i)] * y4[:,int(j)])[:, np.newaxis]
                yout = np.hstack((yout,y5))
    
    if usesine:
        for k in np.arange(1,11):
            yout = np.hstack((yout,np.hstack((np.sin(k*x), np.cos(k*x)))))

    return yout

def hank_modes(data, hank_len):
    H = hankel(data[:hank_len],data[hank_len:]) 
    U, S, Vh = svd(H, full_matrices=False) 
    V = Vh.transpose()
    
    # make this hank_len by len(data)-hank_len
    # full_matricies=False is 'econ'. Is Vh transposed like matlab? S is returned as a vector, not matrix.
    beta = len(H)/len(H[0])
    thresh = opt_SVHT_coef(beta,0) * np.median(S)
    return U,S,V,thresh

def regress(V, rmax=6, order=3, usesine=False):
    r = sum(S>thresh) # length(sigs[sigs>thresh])
    r = min(rmax,r)

    # Compute derivatives
    dt = 0.678 # From the spline interpolation, done in matlab
    dx = 1/(12*dt) * (V[:-4,:r] - 8*V[1:-3,:r] + 8*V[3:-1,:r] - V[4:,:r])
    # finite difference method
    x = V[2:-3,:r]

    # Make Theta

    theta = thetaGen(x,order,usesine) # has to spit out a numpy array
    theta = normalize(theta, axis=0)  # columnwise normalization with euclidean norm
    Xi = lstsq(theta, dx[:len(theta),:]) # theta\dx(:len(theta),:) = linalg.solve(a, b) if a is square, linalg.lstsq(a, b) otherwise
    Xi = Xi[0]
    Xi = normalize(Xi)
    A = Xi[1:r,:r-1].transpose() # transposes a numpy array
    B = Xi[r+1,:r-1][:, np.newaxis] #.transpose() #   :, np.newaxis]
    
    return x, A, B, r, dt

def havok(data, hank_len=100, order=3, rmax=10, usesine=False, forceThresh=10**-5):
	U,S,V,thresh = hank_modes(data, hank_len)
	x, A, B, r, dt = regress(V, rmax, order, usesine) 

	L = np.arange(0,len(x))
	sys = ss(A,B,np.identity(r-1),0*B)
	yout, T, xout = lsim(sys,x[L,r-1],dt*L,x[0,:r-1])

	fig = plt.figure()
	plt.plot(np.arange(dt,len(data[0])*dt,dt),data[0,:])
	plt.title('Time Series')

	fig = plt.figure()	
	plt.plot(np.arange(dt,len(data)*dt,dt),data, 'b')
	plt.plot(T,yout[:,0], 'r')
	plt.title('Time Series, Blue, and System Solution, Red')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(0,len(V[3:,0])),V[3:,0],label='Embedded Pressure Attractor')
	plt.plot(data[0,3:],label='Time Series')
	plt.title('Embedded Pressure Attractor')
	ax.legend()

	fig = plt.figure()
	for i in U[:,:r].transpose():
	    plt.plot(i)

	inds = V[75:,r]**2 > 10**-5
	inds = [i for i, x in enumerate(inds) if x]	
	fig = plt.figure()
	plt.plot(V[:,0], 'b')
	i = 0
	ilocal = []
	while i < len(inds)-1:
		ilocal.append(inds[i])
		if inds[i+1] != inds[i]+1:
		    plt.plot(ilocal,V[ilocal,0],'r')
		    ilocal = []
		i += 1
	plt.title('Time Series, Blue, with Forcing Events, Red')













































