import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
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
#----------------------------plotting Data with LinearDiscriminantAnalysis--------------------------------
for x in range(3):
    for y in range(x+1,3):
        #get Data from table
        petal=np.array([table['petalDims'][x],table['petalDims'][y]]).reshape(100,2)
        varieties=np.array([table['variety'][x],table['variety'][y]]).flatten()
        #train classificator
        clf = LinearDiscriminantAnalysis()
        clf.fit(petal,varieties)
        #find the coef of the line
        w=clf.coef_[0]
        a=-w[0]/w[1]
        xx=np.linspace(min(petal[:,0]),max(petal[:,0]))
        yy0=a*xx-clf.intercept_[0]/w[1]
        #plot
        plt.figure()
        plt.plot(xx,yy0,'r-',label='classification')
        plt.plot(petal[0:49,0],petal[0:49,1],'bo',label=varieties[0])
        plt.plot(petal[50:99,0],petal[50:99,1],'go',label=varieties[99])
        plt.xlabel('petal.length')
        plt.ylabel('petal.width')
        plt.title('Linear Discriminant\n%s VS %s'%(varieties[0],varieties[99]))
        plt.legend()
plt.show()

#----------------------------plotting Data with QuadraticDiscriminantAnalysis--------------------------------
for x in range(3):
    for y in range(x+1,3):
        #get Data from table
        petal=np.array([table['petalDims'][x],table['petalDims'][y]]).reshape(100,2)
        varieties=np.array([table['variety'][x],table['variety'][y]]).flatten()
        #train classificator
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(petal,varieties)

#----------------------------plotting Data with LogisticRegression--------------------------------
for x in range(3):
    for y in range(x+1,3):
        #get Data from table
        petal=np.array([table['petalDims'][x],table['petalDims'][y]]).reshape(100,2)
        varieties=np.array([table['variety'][x],table['variety'][y]]).flatten()
        #train classificator
        clf = LogisticRegression()
        clf.fit(petal,varieties)
        #find the coef of the line
        w=clf.coef_[0]
        a=-w[0]/w[1]
        xx=np.linspace(min(petal[:,0]),max(petal[:,0]))
        yy0=a*xx-clf.intercept_[0]/w[1]
        #plot
        plt.figure()
        plt.plot(xx,yy0,'r-',label='classification')
        plt.plot(petal[0:49,0],petal[0:49,1],'bo',label=varieties[0])
        plt.plot(petal[50:99,0],petal[50:99,1],'go',label=varieties[99])
        plt.xlabel('petal.length')
        plt.ylabel('petal.width')
        plt.title('Logistic Regression\n%s VS %s'%(varieties[0],varieties[99]))
        plt.legend()
plt.show()

#----------------------------plotting Data with SVC--------------------------------
for x in range(3):
    for y in range(x+1,3):
        #get Data from table
        petal=np.array([table['petalDims'][x],table['petalDims'][y]]).reshape(100,2)
        varieties=np.array([table['variety'][x],table['variety'][y]]).flatten()
        #train classificator
        clf = SVC(kernel='linear')
        clf.fit(petal,varieties)
        #find the coef of the line
        w=clf.coef_[0]
        a=-w[0]/w[1]
        xx=np.linspace(min(petal[:,0]),max(petal[:,0]))
        yy0=a*xx-clf.intercept_[0]/w[1]
        #plot
        plt.figure()
        plt.plot(xx,yy0,'r-',label='classification')
        plt.plot(petal[0:49,0],petal[0:49,1],'bo',label=varieties[0])
        plt.plot(petal[50:99,0],petal[50:99,1],'go',label=varieties[99])
        plt.xlabel('petal.length')
        plt.ylabel('petal.width')
        plt.title('SVC\n%s VS %s'%(varieties[0],varieties[99]))
        plt.legend()
plt.show()