# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:37:46 2017

@author: MONIK RAJ
"""

import pandas as pd
import numpy as np

data = pd.read_csv("D:/ex1data1.csv")
Y = data["Profit"]
del data["Profit"]
bias = pd.Series(1,index=range(len(Y))) 
data["Bias"] = bias
'''
Next four lines is to bring bias column to the first since bias column is X0 and previously it is added to the end of the dataframe
'''
Header_X_Bias = list(data.columns.values)
Header_X_Bias = Header_X_Bias[:-1]
Header_X_Bias.insert(0,"Bias")
data = data[Header_X_Bias]

X = np.array(data)
Y = np.array(Y)
Y = Y.reshape(len(Y),1)
Theta = [0,0]
Theta = np.array(Theta)
Theta = Theta.reshape(2,1)

'''
Now X is (97,2)
Y is (97,1)
Theta is (2,1)

So XTheta = Y
   (97,2)(2,1) = (97,1)
'''

def cost(X,Y,Theta):
    Hypothesis = np.dot(X,Theta)
    Error = Y - Hypothesis
    #Matrix method for calculating Cost
    Cost = np.dot(Error.T,Error)/(2*len(Error))
    '''
    #The loop method for calculatin Cost
    for i in range(0,len(Error)):
        Cost = Cost + Error[i]*Error[i]
    Cost = Cost/(2*len(Error))
    '''
    return Cost[0][0]

alpha = 0.01
Iterations = 1500

Cost_History = []
Theta_History = []

A=[]#For plot
B=[]#For plot
C=[]#For plot
D = []#For Plot
def gradient(X,Y,Theta,Iterations,alpha):
    for i in range(Iterations):
        Loss = Y - np.dot(X,Theta) + (np.dot(Theta.T,Theta)*0.001) #L2 Regularization added
        Cost = cost(X,Y,Theta)
        Loss = Loss*(-1)
        dJ = (np.dot(X.T,Loss)*2)/len(Y)#Calculating Partial differentiation of Cost function
        Cost_History.append(Cost)
        Theta_History.append(Theta)
        
        A.append(Theta[0][0])#For plot
        B.append(Theta[1][0])#For plot
        C.append(Cost)#For plot
        D.append([Theta[0][0],Theta[1][0],Cost])
        Theta = Theta - (alpha*dJ) #New Theta
    return Theta

Theta_Iterated = gradient(X,Y,Theta,Iterations,alpha)

Theta_Normal = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)

'''
Below is the code for plotting 3d graph
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
Xp = A
Yp = B
Xp, Yp = np.meshgrid(Xp, Yp)
Zp = C

# Plot the surface.
surf = ax.plot_surface(Xp, Yp, Zp, cstride=90, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(2.01, 34.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
