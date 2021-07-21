import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing Data
data=pd.read_csv("iris.csv")
#dataFrame=data.to_numpy().T
#print(np.corrcoef(dataFrame))


#ploting data
l=data.columns.size-1
for x in range(l)[::-1]:
    for y in range(x+1,l)[::-1]:
        plt.figure()
        plt.title('%s & %s' %(data.columns[x],data.columns[y]))
        plt.xlabel(data.columns[x])
        plt.ylabel(data.columns[y])
        plt.plot(data.loc[0:49][data.columns[x]],data.loc[0:49][data.columns[y]],'bo',label='Setosa')
        plt.plot(data.loc[50:99][data.columns[x]],data.loc[50:99][data.columns[y]],'ro',label='Versicolor')
        plt.plot(data.loc[100:149][data.columns[x]],data.loc[100:149][data.columns[y]],'co',label='Virginica')
        plt.legend()
plt.show()
