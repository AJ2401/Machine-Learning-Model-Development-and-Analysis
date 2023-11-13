# Developijng K-Means Algorithm  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Columns of the data - set
# 1.area A, 
# 2. perimeter P, 
# 3. compactness C = 4*pi*A/P^2, 
# 4. length of kernel,
# 5. width of kernel,
# 6. asymmetry coefficient
# 7. length of kernel groove.
# 8. Class 
def draw_plots(df,cols):
   
   for X in range(len(cols)-1):
      for Y in range(X+1,len(cols)-1):
         x_label = cols[X]
         y_label = cols[Y]
         Title = str(x_label) + " VS " + str(y_label)
         sns.scatterplot(x = x_label,y = y_label,data = df,hue = "class").set(title = Title)
         address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/Clustering_Plots_All_Parameter/Plot_" + Title + ".jpg"
         plt.savefig(address)
   
   return

# def draw_outputs(df,col1,col2):
   
#    x_label = col1
#    y_label = col2
#    Title = str(x_label) + " VS " + str(y_label)
#    sns.scatterplot(x = x_label,y = y_label,data = df,hue = "class").set(title = Title)
#    address = "/Users/abhishekjhawar/Desktop/Project/AI/Programs/Output/K-Mean_Clustering_Output/plot_output_" + Title + ".jpg"
#    plt.savefig(address)
   
#    return


def K_Mean_Model(data,k):
   
   Model = KMeans(n_clusters = k).fit(data)
   predictions = Model.labels_
   
   return predictions

 
def accuracy(predictions,actual):
   
   hits = 0
   miss = 0
   for i in range(len(actual)):
      if predictions[i] == actual[i]:
         hits += 1
      else:
         miss += 1
   Accuracy = (hits/len(actual)) * 100
   Loss = (miss/len(actual)) * 100 
   
   return Accuracy,Loss


def main():
   
   cols = ["area",'perimeter',"compactness","length_of_kernel","width_of_kernel","asymmetry_coefficient","length_of_kernel_groove","class"]
   df = pd.read_csv("/Users/abhishekjhawar/Desktop/Project/AI/Programs/seeds_dataset.txt",names = cols,sep = "\s+")
   print(df.head())
   
   print("\nPlotting All the Plots regarding all the columns to understand the correlation between the variables/factors \n")
   draw_plots(df,cols)
   
   print("\n\n")
   count = 0
   min_model_loss = float('inf')
   min_loss_model = None
   max_model_accuracy = 0.00
   max_accuracy_model = None
   
   for x1 in range(len(cols)-1):
      for x2 in range(x1+1,len(cols)-1):
         print(f"\nThe Model {count} is of {cols[x1]} vs {cols[x2]}")
         data = df[[cols[x1],cols[x2]]].values
         predictions = K_Mean_Model(data,k = 3)
         Accuracy,Loss = accuracy(predictions,np.array(df["class"]))
         print(f"\nAccuracy of the Model {count} is : {Accuracy}\nLoss of the Model{count} is : {Loss}\n")
         if Accuracy > max_model_accuracy:
            max_model_accuracy = Accuracy
            max_accuracy_model = count
         if Loss < min_model_loss:
            min_model_loss = Loss
            min_loss_model = count
         count += 1
   
   print("\n--------------------------------------------CONCLUSION-----------------------------------------------------------\n")
   print(f"\nThe Most Accurate Model is : {max_accuracy_model} with Accuracy of : {max_model_accuracy}\nThe Minimum - Loss Model is : {min_loss_model} with Loss of :{min_model_loss}\n\n")


main()

