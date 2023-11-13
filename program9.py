# Code for Linear Regression for one independent variable
import copy
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def GetVariables(df,y_label,x_labels = None):
   
   df = copy.deepcopy(df)
   if x_labels is None:
      X = df[[c for c in df.columns if c != y_label]].values
   else:
      if len(x_labels) == 1:
         X = df[x_labels[0]].values.reshape(-1,1)
      else:
         X = df[x_labels].values.reshape(-1,1)
   Y = df[y_label].values.reshape(-1,1)
   data = np.hstack((X,Y))
   
   return data,X,Y


def draw_plots(df):
   
   for label in df.columns[1:]:
      if label == "Rented_Bike_Count":
         continue
      else:
         plt.figure(figsize = (10,7))
         plt.scatter(df[label],df["Rented_Bike_Count"])
         plt.xlabel(label)
         plt.ylabel("Rented_Bike_Count")
         title1 = str(label) + " VS Rented_Bike_Count"
         plt.title(title1)
         address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Linear_Regression_1/plot " + label + ".jpg"
         plt.savefig(address)
         plt.close()      
   return

def Plot_Best_Fit_Line(Model,x_train,y_train):
   
   y_values = np.array(x_train).reshape(-1,1)
   predictions_y = Model.predict(y_values)
   predictions_y = np.array(predictions_y).reshape(-1,1)
   plt.figure(figsize = (10,7))
   plt.scatter(x_train,y_train,label = "Data",color = "blue")
   plt.plot(y_values,predictions_y,label = "BEST - FIT LINE",color = "red",linewidth = 3)
   plt.xlabel = "Solar_Radiation"
   plt.ylabel = "Number of Bikes"
   plt.title("BEST - FIT - LINE CONSTRUCTION")
   plt.show()
   plt.close()
   
   return 

def Accuracy(Model,x_test,y_test):
   
   predictions_y= Model.predict(x_test)
   print("\n--------------------------------------------- Library Function -----------------------------------------------------------\n")
   print(f"\nThe Accuracy of the Linear Regression Model (R^2) is  : {Model.score(x_test,y_test)}")
   hits = 0
   miss = 0
   for i in range(len(y_test)):
      if (y_test[i] - predictions_y[i])/y_test[i] <= 0.10:
         hits += 1
      else:
         miss += 1
   
   hit_ratio = hits/len(y_test)
   miss_ratio = miss/len(y_test)
   print("\n--------------------------------------------- Own Accuracy Function -----------------------------------------------------------\n")
   print(f"\nThe Number of HITS are : {hits}\nThe HIT - RATIO is : {hit_ratio}\nThe Number of MISSES are : {miss}\nThe MISS - RATIO is : {miss_ratio}\n\n")
   
   return

def Linear_Regression_Model(x_train,y_train,x_valid,y_valid,x_test,y_test):
   
   Model = LinearRegression()
   Model.fit(x_train,y_train)
   print(f"\nThe Coefficients of the Model is : {Model.coef_}\nThe Y-Intercept is : {Model.intercept_}")
   accuracy = Model.score(x_test,y_test)
   print(f"\nThe Goodness of FIT (R^2),Basically accuracy of the Model Before Validation set is : {accuracy}\n")
   Model.fit(x_valid,y_valid)
   accuracy = Model.score(x_test,y_test)
   print(f"\nThe Goodness of FIT (R^2),Basically accuracy of the Model After Validation set is : {accuracy}\n")
   return Model
   
   
def main():
   
   cols = ["Date","Rented_Bike_Count","Hour","Temperature","Humidity","Wind_speed","Visibility","Dew_point_temperature","Solar_Radiation","Rainfall","Snowfall","Seasons","Holiday","Functional"]
   df = pd.read_csv("/Users/abhishekjhawar/Desktop/Project/AI/Programs/SeoulBikeData.csv")
   df.columns = cols
   df["Functional"] = (df["Functional"] == "Yes").astype(int)
   df = df.drop(['Date',"Holiday","Seasons","Hour"],axis = 1)
   print(df.head())
   
   draw_plots(df)
   
   train,valid,test = np.split(df.sample(frac = 1),[int(0.60 * len(df)),int(0.80 * len(df))])
   print("\n--------------------------------------Training Data Set---------------------------------------------\n")
   print(f"The Number of Functional Bikes are : {np.sum(train['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are  : {np.sum(train['Functional'] == 0)}\n")
   print("\n--------------------------------------Validating Data Set---------------------------------------------\n")
   print(f"The Number of Functional Bikes are : {np.sum(valid['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are  : {np.sum(valid['Functional'] == 0)}\n")
   print("\n--------------------------------------Testing Data Set---------------------------------------------\n")
   print(f"The Number of Functional Bikes are : {np.sum(test['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are  : {np.sum(test['Functional'] == 0)}\n")
   
   Train,x_train,y_train = GetVariables(train,"Rented_Bike_Count","Solar_Radiation")
   Valid,x_valid,y_valid = GetVariables(valid,"Rented_Bike_Count","Solar_Radiation")
   Test,x_test,y_test = GetVariables(test,"Rented_Bike_Count","Solar_Radiation")
   
   print("\n-------------------------------------- Modified Training Data Set---------------------------------------------\n")
   print(len(Train))
   print(f"The Number of Bikes are : \n {y_train}\n")
   print("\n-------------------------------------- Modified Validation Data Set---------------------------------------------\n")
   print(len(Valid))
   print(f"The Number of Bikes are :\n{y_valid}\n")
   print("\n-------------------------------------- Modified Testing Data Set---------------------------------------------\n")
   print(len(Test))
   print(f"The Number of Bikes are :\n{y_test}\n")
   
   Model = Linear_Regression_Model(x_train,y_train,x_valid,y_valid,x_test,y_test)
   Plot_Best_Fit_Line(Model,x_train,y_train)
   Accuracy(Model,x_test,y_test)
   
main()

