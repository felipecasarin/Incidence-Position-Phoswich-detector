from __future__ import division 
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
from numpy import cross, eye, dot
import ImageModelClass as IMC
import ExpPat as EX
import ExpReCal as ERC
import scipy as sp
import Fits as ft
from scipy.optimize import least_squares
from array import array
from scipy import linalg
import pandas as pd
import openpyxl
import os




#eps0=1.
#Ybg0=1.E-4


# tab_eps0 = "./par_eps_Mean_inc_invl_2reflex_switch.txt"
tab_eps0 = "outfiles_pos_calc_0903_pixdim3/parameters/par_eps_09_03_pixdim3.txt"
#tab_eps0 = "./par_eps_test.txt"
eps0 = np.loadtxt(tab_eps0)

#tab_Ybg0 = "./par_Ybg0_2.txt"
#Ybg0 = np.loadtxt(tab_Ybg0)

#Import of calibration parameters b calculated by fitb() from Fits.py
tab_b="outfiles_pos_calc_0903_pixdim3/parameters/par_b_09_03_pixdim3.txt"
#tab_b="./par_b_Mean_inc_invl_2reflex_switch.txt"
#tab_b="./par_b_test.txt"

b=np.loadtxt(tab_b)
#b=np.array([1,1,1,1,1,1,1,1])

#tab_c="./par_c_Mean_inc_invl_2reflex_switch.txt"
tab_c="outfiles_pos_calc_0903_pixdim3/parameters/par_c_09_03_pixdim3.txt"
#tab_c="./par_c_test.txt"

c=np.loadtxt(tab_c)

#c = np.array([0,0,0,0,0,0,0,0])

IM=IMC.ImageModel(eps0)

Lpitch = 4.2

ruler=Lpitch*np.array([-1.5,-0.5,0.5,1.5])


####################################################################

#Opens the root file with the data to be processed
data_address = 'collection_files_csv_unfiltered/-4_2.csv'
#data = pd.read_excel(data_address)
data=pd.read_csv(data_address)
print(data.columns)


#Calculates the Pattern Error Function using xguess, yguess and the Yield Data
def fpatxy(X,Y_dat): 
	x=X[0]
	y=X[1]
	f=ft.PatternErrorOffset(x,y,0.,0.,Y_dat)
	return f

def Ytable(xhole,yhole,xoff=0.,yoff=0.,istonorm=1,a=np.ones((4,4))): # theoretical yields calculated using the hole position (xhole,yhole)
	YT=np.zeros((2,4))  
	x= xhole-xoff
	y= yhole-yoff
	print(x,y)        
	Ypm=IM.YpixModel(x,y)*a    
	YT[0][0:4]=IM.YPattern(Ypm,0,istonorm) #lines
	YT[1][0:4]=IM.YPattern(Ypm,1,istonorm) #columns  
	#print('break') 
	print(YT)       
	return YT

def fitpatxy(newExN): # pattern fit of x,y
	# loads model and calib. fit parameters
    IM.eps=np.loadtxt("outfiles_pos_calc_0903_pixdim3/parameters/par_eps_09_03_pixdim3.txt")
    #IM.eps=np.loadtxt("par_eps_test.txt")
    #IM.Ybg=np.loadtxt("par_Ybg0_2.txt")
    xguess=np.sum(ruler*newExN[0][0:4])*3.
    #print('old xguess=', xguess)
    
    #This part changes the value of xguess and yguess in case they lie outside the detector, as this would lead to miscalculations of the x and y coordinate
    if xguess < -10.:
        xguess = -8.
    if xguess > 10.:
        xguess = 8.
        
    #print('new xguess=', xguess)
    
    yguess=np.sum(ruler*newExN[1][0:4])*3.
    #print('old yguess=', yguess)
    
    if yguess < -10.:
        yguess=-8.
    if yguess > 10.:
        yguess = 8.
    
    #print('new yguess=', yguess)
    
	#print("x,y guess:",xguess,yguess)
    X=np.array([xguess,yguess]) # initial x,y guess
    Lmax=25. #Determines the maximum value for x and y
    bdi=np.array([-Lmax,-Lmax]) # lower limits
    bds=np.array([Lmax,Lmax]) # upper limits	
    
    global result
    #Calculates the x and y coordinates of the interaction point of the alpha particle on the phoswich detector
    result = sp.optimize.least_squares(fpatxy,X,bounds=(bdi,bds),args = ([newExN[0:2]]), ftol=1e-08, xtol=1e-08, gtol=1e-08, loss='soft_l1', tr_solver='lsmr')
    #result = sp.optimize.leastsq(fpatxy,X,bounds=(bdi,bds),args = ([newExN[0:2]]), ftol=1e-08, xtol=1e-08, gtol=1e-08, loss='cauchy', tr_solver='lsmr')
    #print(result)
   
    return result.x

def calib(l1,l2,l3,l4,c1,c2,c3,c4):
    
   
    #Calibrated lines
    if l1 > 0:
        calL1 = (l1)*b[3]+c[3]
    else:
        calL1 = l1
    if l2 > 0:
        calL2 = (l2)*b[2]+c[2]
    else:
        calL2 = l2
    if l3 > 0:
        calL3 = (l3)*b[1]+c[1]
    else:
        calL3 = l3
    if l4 > 0:
        calL4 = (l4)*b[0]+c[0]
    else:
        calL4 = l4
    sumL = calL1 + calL2 + calL3 + calL4
    
    
    #Calibrated columns

    if c1 > 0:
        calC1 = (c1)*b[4]+c[4]
    else:
        calC1 = c1
    if l2 > 0:
        calC2 = (c2)*b[5]+c[5]
    else:
        calC2 = c2
    if l3 > 0:
        calC3 = (c3)*b[6]+c[6]
    else:
        calC3 = c3
    if l4 > 0:
        calC4 = (c4)*b[7]+c[7]
    else:
        calC4 = c4
    sumC = calC1 + calC2 + calC3 + calC4
    
    
    #Normalized and calibrated lines
    cL1 = calL1/(sumL)
    cL2 = calL2/(sumL)
    cL3 = calL3/(sumL)
    cL4 = calL4/(sumL)

    
    #Normalized and calibrated columns
    cC1 = calC1/(sumC)
    cC2 = calC2/(sumC)
    cC3 = calC3/(sumC)
    cC4 = calC4/(sumC)


    calb = np.array([cL1,cL2,cL3,cL4,cC1,cC2,cC3,cC4])


    return calb

def pos0(): 

    #Create empty dataframe for data, calibrated data and positions
    output_columns = ['L1','L2','L3','L4','C1','C2','C3','C4','cL1','cL2','cL3','cL4','cC1','cC2','cC3','cC4','X','Y']
    response = pd.DataFrame(columns=output_columns)


    #Gets the variables event from event
    for entryNum in range(0, len(data)):    
        current_data = data.iloc[entryNum]
        
        
        print('-------------------------------')
        L1, L2, L3, L4, C1, C2, C3, C4 = current_data[['L1', 'L2', 'L3', 'L4', 'C1','C2', 'C3', 'C4']].values.T.tolist()


        #if L1>150 and L2>300 and L3>300 and L4>150 and C1>150 and C2> 100 and C3>100 and C4>150:
        # if L1>0 and L2>0 and L3>0 and L4>0 and C1>0 and C2>0 and C3>0 and C4>0:
        # if True:
        if L1+L2+L3+L4>0 and C1+C2+C3+C4>0:
            print(f"entrynum:{entryNum} L1:{L1}// L2:{L2}// L3:{L3}// L4:{L4}// C1:{C1}// C2:{C2}// C3:{C3}// C4:{C4}")
            calb = calib(L1,L2,L3,L4,C1,C2,C3,C4)  

            #Organizes the calibrated values from each line and column on an array holding two vectors (Lines and Columns)
            newExN = np.array([[calb[3],calb[2],calb[1],calb[0]],[calb[4],calb[5],calb[6],calb[7]]])

            #Executes the optimize least squared routine to calculate position
            xypat = fitpatxy(newExN)
            new_row = {'L1':L1,'L2':L2,'L3':L3,'L4':L4,'C1':C1,'C2':C2,'C3':C3,'C4':C4,'cL1':newExN[0][3],'cL2':newExN[0][2],'cL3':newExN[0][1],'cL4':newExN[0][0],'cC1':newExN[1][0],'cC2':newExN[1][1],'cC3':newExN[1][2],'cC4':newExN[1][3],'X':result.x[1],'Y':result.x[0]}
            response = response._append(new_row, ignore_index=True)
    print("It is done.")
    response.to_excel('-4_2_all_entries.xlsx', index=False)

def pos(x,y): 

    data_address = 'collection_files_csv_unfiltered/' + str(x) +'_' + str(y) +'.csv'
    data=pd.read_csv(data_address)

    #Create empty dataframe for data, calibrated data and positions
    output_columns = ['L1','L2','L3','L4','C1','C2','C3','C4','cL1','cL2','cL3','cL4','cC1','cC2','cC3','cC4','X','Y']
    response = pd.DataFrame(columns=output_columns)


    #Gets the variables event from event
    for entryNum in range(0, len(data)):    
        current_data = data.iloc[entryNum]
        
        
        L1, L2, L3, L4, C1, C2, C3, C4 = current_data[['L1', 'L2', 'L3', 'L4', 'C1','C2', 'C3', 'C4']].values.T.tolist()

        #if L1>150 and L2>300 and L3>300 and L4>150 and C1>150 and C2> 100 and C3>100 and C4>150:
        # if L1>0 and L2>0 and L3>0 and L4>0 and C1>0 and C2>0 and C3>0 and C4>0:
        if L1+L2+L3+L4>0 and C1+C2+C3+C4>0:
            calb = calib(L1,L2,L3,L4,C1,C2,C3,C4)
            # if calb[0] > 0.15 and calb[1] > 0.15 and calb[2] > 0.15 and calb[3] > 0.15 and calb[4] > 0.15 and calb[5] > 0.15 and calb[6] > 0.15 and calb[7] > 0.15:
            if True:
                if entryNum % 100 == 0:
                    print('-------------------------------')
                    print(f"entrynum:{entryNum} L1:{L1}// L2:{L2}// L3:{L3}// L4:{L4}// C1:{C1}// C2:{C2}// C3:{C3}// C4:{C4}")  
                #Organizes the calibrated values from each line and column on an array holding two vectors (Lines and Columns)
                newExN = np.array([[calb[3],calb[2],calb[1],calb[0]],[calb[4],calb[5],calb[6],calb[7]]])

                #Executes the optimize least squared routine to calculate position
                xypat = fitpatxy(newExN)
                new_row = {'L1':L1,'L2':L2,'L3':L3,'L4':L4,'C1':C1,'C2':C2,'C3':C3,'C4':C4,'cL1':newExN[0][3],'cL2':newExN[0][2],'cL3':newExN[0][1],'cL4':newExN[0][0],'cC1':newExN[1][0],'cC2':newExN[1][1],'cC3':newExN[1][2],'cC4':newExN[1][3],'X':result.x[1],'Y':result.x[0]}
                response = response._append(new_row, ignore_index=True)
    print("It is done.")
    response.to_excel('outfiles_no_filter_2/' + str(x) + '_' + str(y) + '_pos' + '.xlsx', index=False)


x_span=np.array([-6,-4,-2,0,2,4,6])
y_span=np.array([-6,-4,-2,0,2,4])

# Function to calculate the position of all the files
def calc_all_pos():
    
    for x_value in x_span:
        for y_value in y_span:
            pos(x_value,y_value)
            print(str(x_value) + ',' + str(y_value) + ' is done!')

# Function to calculate the position of all files in directory, general name


