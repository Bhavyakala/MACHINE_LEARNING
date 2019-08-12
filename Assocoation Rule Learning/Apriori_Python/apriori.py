# Apriori

#importing the libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#apriori expects list of lists 
#dataset.iloc[i,:].values
#str(dataset.values[i,j]) for j in range(0,20)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# training apriori dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,min_lift = 3,min_length = 2)    

#visualising

results = list(rules)   
results1 = pd.DataFrame(results) #spyder does not support frozenset objects in variable explorer
print(results)