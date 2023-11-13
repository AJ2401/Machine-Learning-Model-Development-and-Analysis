# Conding Linear Regression with Multiple Variables in Account
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from sklearn.linear_model import LinearRegression

# If x_labels is None then it will see/review all the columns and if we pass specified columns names then it will review that and get the x & y values from that
def GetVariables(df,y_label,x_labels = None):
   # Basically copying the data frame into a new data-frame.
   dataframe = copy.deepcopy(df) 
   if x_labels is None:
      X = dataframe[[c for c in dataframe.columns if c != y_label]].values
   else:
      if len(x_labels)== 1:
         # Reshape making it into 2D matrix
         X = dataframe[x_labels[0]].values.reshape(-1,1)
      else:
         X = dataframe[x_labels].values
   Y = dataframe[y_label].values.reshape(-1,1)
   data = np.hstack((X,Y))
   
   return data,X,Y
# Plotting Best-Fit Line in the scatter plot of data-points
def Best_Fit_Line_Plot(Model,x_train,y_train):
   # Making  a straight line for f1 and f2 features.
   f1 = tf.linspace(-20,80,100)
   f2 = tf.linspace(-20,80,100)
   # Reshaping the f1 and f2 into 2D arrays.
   features = np.hstack((np.array(f1).reshape(-1,1),np.array(f2).reshape(-1,1)))
   # Getting the predicted values.
   predictions_x = Model.predict(features)
   
   # Plotting a Scattered plot of F1 Feature. 
   plt.figure(figsize = (10,8))
   plt.scatter(x_train[:,0],y_train,label = "Data",color = "blue")
   plt.plot(f1,np.array(predictions_x).reshape(-1,1),label = "FIT",color = "red",linewidth = 3)
   plt.title("Bikes Vs Temperature")
   plt.xlabel = "Temperature"
   plt.ylabel = "Number of Bikes"
   plt.legend()
   plt.show() 
    
   # Plotting a Scattered plot of F2 Feature. 
   plt.figure(figsize = (10,8))
   plt.scatter(x_train[:,1],y_train,label = "Data",color = "blue")
   plt.plot(f2,np.array(predictions_x).reshape(-1,1),label = "FIT",color = "red",linewidth = 3)
   plt.title("Bikes Vs Solar_Radiations")
   plt.xlabel = "Solar_Radiations"
   plt.ylabel = "Number of Bikes"
   plt.legend()
   plt.show() 
   
   return

# The Function defines the Model and Fits the data as well as evaluates the model by using R^2 metrics.
def Linear_Regression(x_train,y_train,x_valid,y_valid,x_test,y_test):
   
   Model = LinearRegression()
   Model.fit(x_train,y_train)
   
   # Printing the Coefficients
   print(f"The Model Coefficients are : {Model.coef_}\nThe Model intersepts are : {Model.intercept_}")
   # Model Accuracy Check before fitting the validation set
   accuracy = Model.score(x_test,y_test)
   print(f"The Model Before Validation Data Set Accuracy is : {accuracy}\n\n")
   
   Model.fit(x_valid,y_valid)
   # Now checking the accuracy after fitting the validation set into our model
   accuracy = Model.score(x_test,y_test)
   print(f"The Model After Validation Data Set Accuracy is : {accuracy}\n\n")
   
   return Model

# Building Scatter plots for Each Features which helps us to choose one of them for Model Creation. 
def draw_plots(df):
   
   for label in df.columns[1:]:
      plt.figure(figsize = (10,7))
      plt.scatter(df[label],df['Rented_Bike_Count'])
      plt.title(label)
      plt.xlabel(label)
      plt.ylabel("Rented_Bike_Count")
      plt.legend()
      # plot jpg image is getting saved in the address 
      address = "Programs/Output/Linear_Regression/Plot_" + label + ".jpg"
      plt.savefig(address)
      
      return
      
def Accuracy(Model,x_test,y_test):
   
   # features = np.hstack((np.array(x_train[:,0]).reshape(-1,1),np.array(x_train[:,1]).reshape(-1,1)))
   # R^2 refects that how the model has fitted the data.
   predictions_y = Model.predict(x_test)
   print("\n---------------------------------------------------------------LIBRARY FUNCTION------------------------------------------------------------------------\n")
   print(f"\n\nAccuracy of the Linear Regression Model By Library Function is (R^2) : {Model.score(x_test,y_test)}\n\n")
   
   hits = 0
   miss = 0
   for i in range(len(x_test)):
      if (y_test[i] - predictions_y[i])/y_test[i] <= 0.10:
         hits += 1
      else:
         miss += 1
         
   hits_rate = (hits/len(x_test)) * 100
   miss_rate = (miss/len(x_test)) * 100
   # My custom function measures how often the model's predictions are within a certain range of the actual values. 
   print("\n---------------------------------------------------------------OWN FUNCTION------------------------------------------------------------------------\n")
   print(f"\nThe Number of HITS are : {hits}\nThe HIT-RATIO is : {hits_rate}\nThe Number of MISS are : {miss}\nThe MISS-RATIO is : {miss_rate}\n")
   
   return

def main():
   
   cols = ["Date","Rented_Bike_Count","Hour","Temperature","Humidity","Wind_speed","Visibility","Dew_point_temperature","Solar_Radiation","Rainfall","Snowfall","Seasons","Holiday","Functional"]
   df = pd.read_csv("/Users/abhishekjhawar/Desktop/Project/AI/Programs/SeoulBikeData.csv")
   # Dropping some of the columns which are not needed that much
   df.columns = cols
   df = df.drop(['Date',"Holiday","Seasons","Hour"],axis = 1)
   df["Functional"] = (df["Functional"] == "Yes").astype(int)
   print(df.head())
   
   
   draw_plots(df)
   # By reviewing all the plots,Ploar_Solar_Radiations plot is the most scattered and best one to choose for linear-regression analysis.
   # So now droping all the unnecessary columns.
   # df = df.drop(["Temperature","Humidity","Wind_speed","Visibility","Dew_point_temperature"],axis = 1)
      
   train,valid,test = np.split(df.sample(frac = 1),[int(0.60 * len(df)),int(0.80 * len(df))])
   print("\n--------------------------------------Training Data Set---------------------------------------------\n")
   print(train)
   print(f"The Number of Functional Bikes are : {np.sum(train['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are  : {np.sum(train['Functional'] == 0)}\n")
   print("\n--------------------------------------Validation Data Set---------------------------------------------\n")
   print(valid)
   print(f"The Number of Functional Bikes are :{np.sum(valid['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are : {np.sum(valid['Functional'] == 0)}\n")
   print("\n--------------------------------------Testing Data Set---------------------------------------------\n")
   print(test)
   print(f"The Number of Functional Bikes are :{np.sum(test['Functional'] == 1)}\n")
   print(f"The Number of Non-Functional Bikes are : {np.sum(test['Functional'] == 0)}\n")
   
   # Here I will try to get the x and the y from the dataframe
   Train,x_train,y_train = GetVariables(train,"Rented_Bike_Count",x_labels=["Temperature","Solar_Radiation"])
   Valid,x_valid,y_valid = GetVariables(valid,"Rented_Bike_Count",x_labels=["Temperature","Solar_Radiation"])
   Test,x_test,y_test = GetVariables(test,"Rented_Bike_Count",x_labels=["Temperature","Solar_Radiation"])
   
   print("\n-------------------------------------- Modified Training Data Set---------------------------------------------\n")
   print(len(Train))
   print(f"The Number of Bikes are : \n {y_train}\n")
   print("\n-------------------------------------- Modified Validation Data Set---------------------------------------------\n")
   print(len(Valid))
   print(f"The Number of Bikes are :\n{y_valid}\n")
   print("\n-------------------------------------- Modified Testing Data Set---------------------------------------------\n")
   print(len(Test))
   print(f"The Number of Bikes are :\n{y_test}\n")
   
   Model = Linear_Regression(x_train,y_train,x_valid,y_valid,x_test,y_test)
   print("\n\nPlotting the Best Fit Line : \n\n")
   Best_Fit_Line_Plot(Model,x_train,y_train)
   Accuracy(Model,x_test,y_test)
   
   return
  
main()