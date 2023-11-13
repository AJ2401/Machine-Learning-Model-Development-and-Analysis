# Coding the Naive's Bayes Theorem 
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler

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

def accuracy(y_test,y_predictions):
   hits = 0
   miss = 0
   
   for i in range(len(y_test)):
      if y_test[i] != y_predictions[i]:
         miss += 1
      else:
         hits += 1
   
   hit_ratio = hits/len(y_test)
   miss_ratio = miss/len(y_test)
   print(f"\nThe Number of Hits are : {hits} \nThe Number of Misses are :{miss}\nThe HIT-RATIO is : {hit_ratio*100} % \nThe MISS-RATIO is : {miss_ratio*100} % \n\n")
   
   return

def GModel(x_train,y_train,x_test,y_test):
   #Now we will start the model thing : 
   model = GaussianNB()
   model = model.fit(x_train,y_train)
   y_predictions = model.predict(x_test)
   # The Results are not great enough
   print("\n\n----------------------------- Gaussian Model Report By Library Function ------------------------------------\n\n")
   print(classification_report(y_test,y_predictions))
   
   print("\n\n----------------------------- Gaussian Model Report By Own Function ------------------------------------\n\n")
   print(accuracy(y_test,y_predictions))
   
   return y_predictions

def main():
   cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
         "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

   df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data",names = cols)
   df["Value"] = (df['class'] == 'g').astype(int)
   print(df.head())

   train,valid,test = np.split(df.sample(frac = 1),[int(0.6*len(df)),int(0.8*len(df))])
   print("\n\nTRAINING DATA SET \n\n")
   print(train)
   print("\n\nValid\n\n")
   print(valid)
   print("\n\nTESTING DATA SET\n\n")
   print(test)
   
   train,x_train,y_train = Scaling(train,oversampling=True)
   valid,x_valid,y_valid = Scaling(valid,oversampling=False)
   test,x_test,y_test = Scaling(test,oversampling=False)
   print("\nNew Random Sized Data Set \n")
   print(" ------- Training ---------- ")
   print(f"\nHydrons Count : {np.sum(y_train == 0)}\n")
   print(f"Gamma Count : {np.sum(y_train == 1 )}\n")
   print(" --------- Validating ------------ ")
   print(f"\nHydrons Count : {np.sum(y_valid == 0)}\n")
   print(f"Gamma Count : {np.sum(y_valid == 1 )}\n")
   print(" ----------- Testing --------------- ")
   print(f"\nHydrons Count : {np.sum(y_test == 0)}\n")
   print(f"Gamma Count : {np.sum(y_test == 1 )}\n")
   
   #model function
   st1 = time.time()
   y_predictions = GModel(x_train,y_train,x_test,y_test)
   end1 = time.time()
   print(f"\n\nTime Taken by the Model Function is : {end1-st1:.6f} ms\n\n")
   
main()
