import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def Scaling(df,oversampling=False):
   x = df[df.columns[:-2]].values 
   y = df[df.columns[-1]].values
   
   scaler = StandardScaler()
   x = scaler.fit_transform(x)
   
   if oversampling:
      ros = RandomOverSampler()
      x,y = ros.fit_resample(x,y)
   
   data = np.hstack((x,np.reshape(y,(len(y),1))))
   
   return data,x,y
    
def Accuracy(y_test,y_predictions):
   hits = 0
   miss = 0 
   for i in range(len(y_test)):
      if y_test[i] != y_predictions[i]:
         miss += 1
      else:
         hits += 1
   hit_ratio = (hits/len(y_test)) * 100
   miss_ratio = (miss/len(y_test)) * 100
   
   print(f"\n\tThe Number of Hits are : {hits} \n\tThe Number of Misses are : {miss}\n\tThe Hit Ratio is : {hit_ratio} % \n\tThe Miss Ratio is : {miss_ratio} % \n")


def Logistic_Regression(x_train,y_train,x_test,y_test):
   model = LogisticRegression()
   model = model.fit(x_train,y_train)
   
   y_predictions = model.predict(x_test)
   print("\n\n------------------------- Logistic Model Report by Library Function ---------------------------------------------\n\n")
   print(classification_report(y_test,y_predictions))
   print("\n\n------------------------- Logistic Model Report by Own Function ---------------------------------------------\n\n")
   Accuracy(y_test,y_predictions)
   
   return 
   
def main():
   cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
         "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
   df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data",names = cols)
   df["Value"] = (df["class"] == 'g').astype(int)
   print(df.head())
   
   train,valid,test = np.split(df.sample(frac=1),[int(0.60 * len(df)),int(0.80 * len(df))])
   print("\n-------------------------- Training Data Set-------------------------------------------------\n")
   print(train)
   print("\n-------------------------- Valid Data Set-------------------------------------------------\n")
   print(valid)
   print("\n-------------------------- Testing Data Set-------------------------------------------------\n")
   print(test)
   
   train,x_train,y_train = Scaling(train,oversampling = True)
   valid,x_valid,y_valid = Scaling(valid,oversampling = False)
   test,x_test,y_test = Scaling(test,oversampling = False)
   print("\n\nAfter Random Scaling of Data Set \n\n")
   print("\n-------------------------- Training Data Set-------------------------------------------------\n")
   print(np.sum(y_train == 1))
   print("\n-------------------------- Valid Data Set-------------------------------------------------\n")
   print(np.sum(y_valid == 1))
   print("\n-------------------------- Testing Data Set-------------------------------------------------\n")
   print(np.sum(y_test == 1))
   
   st1 = time.time()
   Logistic_Regression(x_test,y_test,x_test,y_test)
   end1 = time.time()
   print(f"\n\nTime Taken by the Model is : {end1-st1:.6f} ms\n\n")
   
main()