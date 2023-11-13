import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
   
   print(f"\n\tThe Number of Hits are : {hits}\n\tThe Number of Miss are : {miss}\n\tThe HIT-RATIO is : {hit_ratio} %\n\tThe MISS-RATIO is : {miss_ratio} %\n")
   
   return

def Support_Vector_Machine(x_train,y_train,x_test,y_test):
   
   model = SVC()
   model = model.fit(x_train,y_train)
   
   predictions = model.predict(x_test)
   print("\n\n-------------------------------------------Support Vector Machine Model Report by LIBRARY FUNCTION ----------------------------------------\n\n")
   print(classification_report(y_test,predictions))
   print("\n\n-------------------------------------------Support Vector Machine Model Report by OWN FUNCTION ----------------------------------------\n\n")
   Accuracy(y_test,predictions)
   
   return 

def main():
   cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1","fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
   df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data",names=cols)
   df["Value"] = (df['class'] == 'g').astype(int)

   train,valid,test = np.split(df.sample(frac = 1),[int(0.6 * len(df)),int(0.8 * len(df))])
   
   print("\n---------------------------------Training Data Set---------------------------------\n")
   print(train)
   print("\n---------------------------------Validating Data Set---------------------------------\n")
   print(valid)
   print("\n---------------------------------Testing Data Set---------------------------------\n")
   print(test)
   
   train,x_train,y_train = Scaling(train,oversampling=True)
   valid,x_valid,y_valid = Scaling(valid,oversampling=False)
   train,x_test,y_test = Scaling(test,oversampling=False)
   print("\nAfter Random Sampling/Scaling\n")
   print("\n---------------------------------Training Data Set---------------------------------\n")
   print(np.sum(y_train == 1))
   print("\n---------------------------------Validating Data Set---------------------------------\n")
   print(np.sum(y_valid == 1))
   print("\n---------------------------------Testing Data Set---------------------------------\n")
   print(np.sum(y_test == 1))
   
   st1 = time.time()
   Support_Vector_Machine(x_train,y_train,x_test,y_test)
   end1 = time.time()
   
   print(f"\nThe Time Taken by the Model is : {end1-st1:.6f} ms \n")
    
main()