# Coding/Developing a Linear Regression Model using a Neural Network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from sklearn.metrics import r2_score

def draw_plots(df):
   
   for label in df.columns[1:]:
      if label == "Rented_Bike_Count":
         continue
      else:
         plt.figure(figsize = (10,7))
         plt.scatter(df[label],df["Rented_Bike_Count"],color = "blue")
         plt.xlabel = label
         plt.ylabel = "Rented_Bike_Count"
         title = label + " VS Rented_Bike_Count"
         plt.title(title)
         address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Linear_Regression_NN/plot_1 " + label + ".jpg"
         plt.savefig(address)
         plt.close()
         
   return 


def GetVariables(df,y_label,x_labels = None):
   
   data = copy.deepcopy(df)
   if x_labels == None:
      X = data[[c for c in df.columns if c != y_label]].values
   else:
      if len(x_labels) == 1:
         X = data[x_labels[0]].values.reshape(-1,1)
      else:
         X = data[x_labels].values.reshape(-1,1)
   Y = data[y_label].values.reshape(-1,1)
   data = np.hstack((X,Y))
   
   return data,X,Y


def plot_best_fit_line(Model,x_train,y_train,iterator):
   
   plt.scatter(x_train,y_train,label = "Data",color = "blue")
   predictions_y = Model.predict(np.array(x_train).reshape(-1,1))
   plt.plot(x_train,predictions_y,label = "FIT",color = "red",linewidth = 3)
   plt.ylabel = "Rented_Bike_Count"
   plt.xlabel = "Solar_Radiation"
   plt.title("Rented_Bike_Count VS Solar_Radiation")
   plt.legend()
   address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Linear_Regression_plot_best_fit_line/plot_ " + str(iterator) + ".jpg"
   plt.savefig(address)
   plt.close()
   
   return 


def plot_history_NN(history,iterator):
   
   plt.plot(history.history['loss'],label = "Loss")
   plt.plot(history.history["val_loss"],label = "Value_Loss")
   plt.xlabel = "Epoch"
   plt.ylabel = "MSE"
   plt.grid(True)
   address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Linear_Regression_NN_Plots/plot_ " + str(iterator) + ".jpg"
   plt.savefig(address)
   plt.close()
   
   return
   
   
def Linear_NN_Model(x_train,y_train,x_valid,y_valid,epochs,learning_rate):
   # Creating a Normalizer layer 
   
   normalizer_data = tf.keras.layers.Normalization(input_shape = (1,),axis = None)
   normalizer_data.adapt(x_train.reshape(-1))
   NN_Model = tf.keras.Sequential(
      [
         normalizer_data,
         tf.keras.layers.Dense(1)
      ]
   )
   NN_Model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),loss = "mean_squared_error")
   history = NN_Model.fit(
      x_train.reshape(-1),y_train,verbose = 0,epochs = epochs ,validation_data = (x_valid,y_valid)
   )
   
   return NN_Model,history
   
   
def main():
   
   cols = ["Date","Rented_Bike_Count","Hour","Temperature","Humidity","Wind_speed","Visibility","Dew_point_temperature","Solar_Radiation","Rainfall","Snowfall","Seasons","Holiday","Functional"]
   df = pd.read_csv("/Users/abhishekjhawar/Desktop/Project/AI/Programs/SeoulBikeData.csv")
   df.columns = cols
   df = df.drop(["Date","Hour","Holiday"],axis = 1)
   df["Functional"] = (df['Functional'] == 'Yes').astype(int)
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
   Test,x_test,y_test  = GetVariables(test,"Rented_Bike_Count","Solar_Radiation")
   
   print("\n-------------------------------------- Modified Training Data Set---------------------------------------------\n")
   print(len(Train))
   print(f"The Number of Bikes are : \n {y_train}\n")
   print("\n-------------------------------------- Modified Validation Data Set---------------------------------------------\n")
   print(len(Valid))
   print(f"The Number of Bikes are :\n{y_valid}\n")
   print("\n-------------------------------------- Modified Testing Data Set---------------------------------------------\n")
   print(len(Test))
   print(f"The Number of Bikes are :\n{y_test}\n")
   
   least_value_loss = float('inf')
   least_value_model = None
   maximum_accuracy = 0.00
   maximum_accuracy_model = None
   epochs = int(input("Enter the Number of Epochs (Epochs : Number of Times the Algorithm should run.)\n"))
   count = 1
   for learning_rate in [0.005,0.001,0.1]:
      print(f"The Learning-Rate of the Model is : {learning_rate}\n\n")
      Model,history = Linear_NN_Model(x_train,y_train,x_valid,y_valid,epochs,learning_rate)
      plot_best_fit_line(Model,x_train,y_train,count)
      plot_history_NN(history,count)
      print(Model.summary())
      count += 1
      loss = Model.evaluate(x_test,y_test)
      if loss < least_value_loss:
         least_value_loss = loss
         least_value_model = count - 1
      predictions = Model.predict(x_test).flatten()
      r_squared = r2_score(y_test, predictions)
      if r_squared > maximum_accuracy:
         maximum_accuracy = r_squared
         maximum_accuracy_model = count - 1
         
   print("\nDone The Model is Successfully Trained !\n")
   print("\n---------------------------------------- Conclusion ----------------------------------------------\n")
   print(f"\n\nThe Least Value Loss Model is : {least_value_model} with loss rate value : {least_value_loss}\n\nThe Maximum Accuracy of the Model is : {maximum_accuracy_model} with Accuracy of : {maximum_accuracy}\n\n")
       

main()
