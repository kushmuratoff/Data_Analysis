import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import *
from sympy import *

data=pd.read_csv('iris.csv')
table={
    'petalDims':[
        data[['petal.length','petal.width']].loc[0:49].to_numpy()
        ,data[['petal.length','petal.width']].loc[50:99].to_numpy()
        ,data[['petal.length','petal.width']].loc[100:149].to_numpy()
    ]
    ,'variety':[
        data[['variety']].loc[0:49].to_numpy()
        ,data[['variety']].loc[50:99].to_numpy()
        ,data[['variety']].loc[100:149].to_numpy()

    ]
    }


#----------------------------plotting Data with Quadratic Discriminant--------------------------------
for i in range(2):
    for j in range(i+1,3):
        #get Data from table
        petal=np.array([table['petalDims'][i],table['petalDims'][j]]).reshape(100,2)
        varieties=np.array([table['variety'][i],table['variety'][j]]).flatten()
        #calculating statistics properties
        m1=np.array([mean(petal[0:49,0]),mean(petal[0:49,1])])
        m2=np.array([mean(petal[50:99,0]),mean(petal[50:99,1])])
        S1=np.cov(petal[0:49].T)
        S2=np.cov(petal[50:99].T)
        #calculating coefs
        A=np.linalg.inv(S1-S2)

        invS2m2=np.matmul(np.linalg.inv(S2),m2)
        invS1m1=np.matmul(np.linalg.inv(S1),m1)
        B=2*(invS2m2-invS1m1)

        m1TinvS1m1=np.matmul(m1.T,invS1m1)
        m2TinvS2m2=np.matmul(m2.T,invS2m2)
        detS1=np.linalg.det(S1)
        detS2=np.linalg.det(S2)
        c=m1TinvS1m1-m2TinvS2m2+np.log(detS1/detS2)
        
        x,y=symbols('x,y')
        y=solve(A[0,0]*x**2+A[1,1]*y**2+2*A[0,1]*x*y+B[0]*x+B[0]*y+c,y)[0]
        xx=np.linspace(min(petal[:,0]),max(petal[:,0]))
        yy=[]
        for t in xx:
            yy=yy+[y.subs(x,t)]
        
        #plot
        plt.figure()
        #plt.plot(xx,yy,'r-',label='classification')
        plt.plot(petal[0:49,0],petal[0:49,1],'bo',label=varieties[0])
        plt.plot(petal[50:99,0],petal[50:99,1],'go',label=varieties[99])
        plt.xlabel('petal.length')
        plt.ylabel('petal.width')
        plt.title('Linear Discriminant\n%s VS %s'%(varieties[0],varieties[99]))
        plt.legend()
plt.show()
