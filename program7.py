# Developing a Neural Network
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
 
def scaling(df,oversampling = False):
   df_x = df[df.columns[:-2]].values
   df_y = df[df.columns[-1]].values
   
   scaler = StandardScaler()
   df_x = scaler.fit_transform(df_x)
   
   if oversampling:
      Ros = RandomOverSampler()
      df_x,df_y = Ros.fit_resample(df_x,df_y)
   
   df = np.hstack((df_x,np.reshape(df_y,(len(df_y),1))))
   
   return df,df_x,df_y 

def draw_plot(history,iterator):
   fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,6))
   ax1.plot(history.history['loss'],label = "Loss")
   ax1.plot(history.history['val_loss'],label = "Loss_Value")
   ax1.set_xlabel("EPOCH")
   ax1.set_ylabel("BINARY COSSENTROPY")
   ax1.legend()
   ax1.grid(True)
   
   ax2.plot(history.history['accuracy'],label = "Accuracy")
   ax2.plot(history.history['val_loss'],label = "Loss_Value")
   ax2.set_xlabel("EPOCH")
   ax2.set_ylabel("BINARY COSSENTROPY")
   ax2.legend()
   ax2.grid(True)
   
   address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Neural_Network_"+str(iterator)+".jpg"
   plt.savefig(address)
   
   
def Neural_Net(x_train,y_train,x_valid,y_valid,number_nodes,epochs,batch_size,learning_rate,drop_propbability):
   # So we use Keras which is an API helps in neural network modelling.
   model = tf.keras.Sequential([
      # Dense : It is the type of Layer
      # relu : Rectified Linear Activation Function.
      # Sigmoid : Function which lies between 0 - 1. 
      # As the Last layer is Sigmoid means the answer will come come between 0 - 1 which we will convert it into 0 or 1 by floor/round function.
      # Here we introduce dropout function which will randomly remove the nodes which will help the model from over-fitting.
   tf.keras.layers.Dense(units = number_nodes,activation = 'relu',input_shape =(10,)),
   tf.keras.layers.Dropout(drop_propbability),
   tf.keras.layers.Dense(units = number_nodes,activation = 'relu'),
   tf.keras.layers.Dropout(drop_propbability),
   tf.keras.layers.Dense(units = 1,activation = "sigmoid")
   ])
   
   model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate),loss = "binary_crossentropy",metrics = ["accuracy"])
   # Tensor Flow keeps the track of it's histrory so we can track that
   history = model.fit(x_train,y_train,epochs = epochs,batch_size = batch_size,validation_split = 0.2,verbose = None)
   
   return model,history

   
def main():
  cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1","fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"] 
  df = pd.read_csv("/Users/abhishekjhawar/Desktop/Project/AI/Programs/magic+gamma+telescope/magic04.data",names = cols)
  df["Value"] = (df["class"] == 'g').astype(int)
  print(df.head())
  
  train,valid,testing = np.split(df.sample(frac = 1),[int(0.60 *len(df)),int(0.80 * len(df))])
  
  print("\n----------------------------------------Training Data Set----------------------------------------------\n")
  print(train)
  print(f"\nNumber of Gamma : {np.sum(train['Value'] == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(train['Value'] == 0)}\n")
  print("\n----------------------------------------Validating Data Set----------------------------------------------\n")
  print(valid)
  print(f"\nNumber of Gamma : {np.sum(valid['Value'] == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(valid['Value'] == 0)}\n")
  print("\n----------------------------------------Testing Data Set----------------------------------------------\n")
  print(testing)
  print(f"\nNumber of Gamma : {np.sum(testing['Value'] == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(testing['Value'] == 0)}\n")
  
  train,x_train,y_train = scaling(train,oversampling = True)
  valid,x_valid,y_valid = scaling(valid,oversampling = False)
  testing,x_test,y_test = scaling(testing,oversampling = False)
  
  print("\n\nAFTER RANDOM SAMPLING\n\n")
  print("\n----------------------------------------Training Data Set----------------------------------------------\n")
  print(f"\nNumber of Gamma : {np.sum(y_train == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(y_train == 0)}\n")
  print("\n----------------------------------------Validating Data Set----------------------------------------------\n")
  print(f"\nNumber of Gamma : {np.sum(y_valid == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(y_valid == 0)}\n")
  print("\n----------------------------------------Testing Data Set----------------------------------------------\n")
  print(f"\nNumber of Gamma : {np.sum(y_test == 1)}\n")
  print(f"\nNumber of Hetrons : {np.sum(y_test == 0)}\n")
  
  print("\nGetting the Model Ready : \n")
  least_value_loss = float('inf')
  least_value_loss_model = None
  maximum_accuracy = 0.00
  maximum_accuracy_model = None
   # Epochs : Basically the Number of Iterations over the data-set. The Number of Times the Algorithm will run over the data-set.
  epochs = int(input("Enter the Number of Epochs : "))
  count = 1
  for number_nodes in [16,32,64,128]:
     for drop_probability in [0,0.2]:
        for learning_rate in [0.005,0.001,0.1]:
           for batch_size in [16,32,64,128]:
              print(f"Number of Nodes : {number_nodes}\nRow Drop Probability : {drop_probability}\nThe Learning Rate is : {learning_rate}\nThe Batch Size : {batch_size}\nThe Epochs Value is : {epochs} \n\n")
              model,history = Neural_Net(x_train,y_train,x_valid,y_valid,number_nodes,epochs,batch_size,learning_rate,drop_probability)
              draw_plot(history,count)
              print(model.summary())
              count += 1
              evaluation = model.evaluate(x_test,y_test)
              if evaluation[0] < least_value_loss:
                 least_value_loss = evaluation[0]
                 least_value_loss_model = model
              if evaluation[1] > maximum_accuracy:
                 maximum_accuracy = evaluation[1]
                 maximum_accuracy_model = model
                 
  print("\nDone The Model is Successfully Trained !\n")
  print("\n---------------------------------------- Conclusion ----------------------------------------------\n")
  print(f"\n\nThe Least Value Loss Model is : {least_value_loss_model} with loss rate value : {least_value_loss}\n\nThe Maximum Accuracy of the Model is : {maximum_accuracy_model} with Accuracy of : {maximum_accuracy}\n\n")
  
main()
