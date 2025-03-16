# -*- coding: cp1252 -*-
from __future__ import division 
from math import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import cross, eye, dot
import ImageModelClass as IMC
import ExpPat as EX
import ExpReCal as ERC
import scipy as sp
from scipy.optimize import least_squares
import sys
import time


#list of experimental data point positions 
xtab=EX.xtab
ytab=EX.ytab
Nx=xtab.size
Ny=ytab.size

ExDataTab=EX.ExDataTab #data: ExDataTab=intable.reshape(Nx,Ny,2,4) 
RelErrorDat=EX.RelErrorDat

a=np.ones((4,4))
b=np.ones((2,4))
c=np.zeros((2,4))
ExN=ERC.normdata(ERC.recaldata(ExDataTab,b,c))

eps0=0.5

def SaveParameters():
	np.savetxt("par_eps.txt",np.array([eps0]))
	np.savetxt("par_a.txt", np.ravel(a))
	np.savetxt("par_b.txt", np.ravel(b))
	np.savetxt("par_c.txt", np.ravel(c))
	return

IM=IMC.ImageModel(eps0)

YpixModel=IM.YpixModel #model: YpixPattern(x,y,ep=eps, a=np.ones((4,4)),istonormalize=1):
YPattern=IM.YPattern	#model: YPattern(Ypm,direction,a=np.ones((4,4)),b=np.ones((2,4)),istonormalize=1)


def Ytable(xoff=0.,yoff=0.,istonorm=1,a=np.ones((4,4))): # theoretical yields
	YT=np.zeros((Nx,Ny,2,4))
	for ix in range(0,Nx):
		for iy in range(0,Ny):
			x=xtab[ix]-xoff
			y=ytab[iy]-yoff
			Ypm=YpixModel(x,y)*a
			YT[ix][iy][0][0:4]=YPattern(Ypm,0,istonorm) #lines
			YT[ix][iy][1][0:4]=YPattern(Ypm,1,istonorm) #columns       
            
	return YT

def errorfunc(xoff,yoff,dTab,YTab): #dTab is the data table after calib.+ renorm. for example
	return YTab-dTab

def PatternError(x,y,LCdat): 
	Ypm=YpixModel(x,y) # theory
	YE=YPattern(Ypm,0)-LCdat[0]
	YE=np.append(YE,YPattern(Ypm,1)-LCdat[1])
	return YE

def PatternErrorOffset(x,y,xoff,yoff,LCdat): 
	Ypm=YpixModel(x-xoff,y-yoff) # theory
	YE=YPattern(Ypm,0)-LCdat[0]
	YE=np.append(YE,YPattern(Ypm,1)-LCdat[1])
	return YE

	
def fbc(X,Y_dat,Y_sig):
	IM.eps=X[0] # sets model parameters for errorfunc eval
	b=np.reshape(X[1:9],(2,4))
	c=np.reshape(X[9:17],(2,4))
	ExN=ERC.normdata(ERC.recaldata(Y_dat,b,c))# 
	YTab=Ytable(0,0)
	f=np.ravel(errorfunc(0.,0.,ExN,YTab)/abs(Y_sig))[0:328] # drop last experimental point (4l,4c, [328:336])
	return f


def newfitbc(eps0=0.8,b=np.ones((2,4)),c=np.zeros((2,4)),blim=10.,clim=1000.): #blim minimum, xylim min/max both x/y

	# Set saving directory
	save_directory = "./outfiles_pos_calc_0903_pixdim3/parameters/"
	#initial values
	starting_time = time.time()
	print("Initializing Parameters Fitting")
	c.fill(100.)
	X=np.array([])
	X=np.append(X,eps0) # initial eps
	X=np.append(X,np.ravel(b)) # initial b’s
	X=np.append(X,np.ravel(c)) # initial c’s

	#lower limits:
	bdi=np.array([0.])
	bdi=np.append(bdi,np.ones((8))/blim)
	bdi=np.append(bdi,np.ones((8))*(-clim))
	bdi[3]=0.999
	bdi[11]=100. # effectively fixes b,c for this line

	#upper limits:
	bds=np.array([])
	bds=np.append(bds,1.)
	bds=np.append(bds,np.ones((8))*blim)
	bds=np.append(bds,np.ones((8))*(clim))
	bds[3]=1.001
	bds[11]=110.

	result = sp.optimize.least_squares(fbc,X,bounds=(bdi,bds),args = ([ExDataTab,ExN*RelErrorDat]),loss='soft_l1',tr_solver='lsmr',method='trf', verbose=2)

	print(result)
	np.savetxt(f"{save_directory}par_eps_09_03_pixdim3.txt",np.array([result.x[0]]))
	np.savetxt(f"{save_directory}par_b_09_03_pixdim3.txt", result.x[1:9])
	np.savetxt(f"{save_directory}par_c_09_03_pixdim3.txt", result.x[9:17])
	ending_time = time.time()
	print("Elapsed time:",ending_time - starting_time)
	return result.x
