import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
#importing Data
data=pd.read_csv("iris.csv")


l=data.columns.size-1
for x in range(l)[::-1]:
    for y in range(x+1,l)[::-1]:
         #get Data from table
        attributes=data.loc[50:149][[data.columns[x],data.columns[y]]].to_numpy().reshape(100,2)
        classes=data.loc[50:149]['variety'].to_numpy().flatten()
        #train classificator
        clf = LinearDiscriminantAnalysis()
        clf.fit(attributes,classes)
        #find the coef of the line
        w=clf.coef_[0]
        a=-w[0]/w[1]
        xx=np.linspace(min(attributes[:,0]),max(attributes[:,0]))
        yy0=a*xx-clf.intercept_[0]/w[1]
        #plot
        plt.figure()
        plt.title('%s & %s' %(data.columns[x],data.columns[y]))
        plt.xlabel(data.columns[x])
        plt.ylabel(data.columns[y])
        plt.plot(xx,yy0,'r-',label='classification')
        plt.plot(data.loc[50:99][data.columns[x]],data.loc[50:99][data.columns[y]],'ro',label='Versicolor')
        plt.plot(data.loc[100:149][data.columns[x]],data.loc[100:149][data.columns[y]],'co',label='Virginica')
        plt.legend()
plt.show()
