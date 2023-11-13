# Machine-Learning-Model-Development-and-Analysis
# Artificial Intelligence

#### Machine Learning is a Sub-Domain of Computer Science that focuses on algorithms which help a computer to learn and understand from data without explicit programming

### What is Explicit Programming ?
```

When there is a programmer is there and giving/providing instructions and information to the system/software/computer.

```

### What is Artificial Intelligence or AI ?

```

It is a domain of computer Science where the goal of programmers is to enable computers and machines to perform human like tasks and simulate human behaviour.

```

### What is Machine Learning or ML ?

```

It is a subset of Artificial Intelligence that tries to solve a specific problem and make predictions using prior/past data.

```

### What is Data Science ? 

```

Data Science is a field that attempts to draw patterns and data info/insights.
It can be used in ML .
It can also be used to understand the nature of the data , variables or factors and their relationships.

```

### Types of Machine Learning 

---

### 1. Supervised Learning : 

#### Defination :

<em>Supervised Learning uses a training set includes inputs and correct outputs/actual outputs,which allows the model to learn over the time. The Algorithm measures it's accuracy through the loss function,adjusting until the error has been sufficient minimised.</em>

#### Data Set :

1. Uses Label Inputs to train the Model.
2. The data set will contain an input and output.
3. Used to classify data and  predict outcomes accurately.
4. After training the model we can cross check the model because of the data set.


#### Procedure :


Suprevised Learning can be classified into two problems :

   1. Classification :   
   - It uses an Algorithm to accurately assign test data into specific categories.
   - It recognises specific entities within the data-set and draw so vauge conclusions on <em>" How the entities should be labelled "</em> 
   - Comman Algorithms (linear algos) are :

      - Decision Trees
      - K-Nearest Neighbor
      - Random Forest
      - etc .

   2. Regression : 
   - It is used to  understand the realtionships between dependent and independent variables.
    - Comman Algorithms for Regression are :

      - Linear Regression (univariate or multivariate)
      - Logistic Regression 
      - Logarithmic Regression 
      - Polynomic Regression

---

### 2. Unsupervised Learning :

#### Defination :

<em>

- Unsupervised Learning can be achieved using  machine learning algorithms to analyze and cluster unlabeled datasets.

- The Model is a learning based model which can learn new things from each input data set.

- There is not restriction on interpretation.
exploratory data analysis, cross-selling strategies, customer segmentation, and image recognition.

</em>


#### Data Set :

1. The data is not labelled means the data set doesn't have correct ouputs for the inputs.
2. The data set contains only inputs which is clustered/grouped by the machine learning algorithm.
3. Using this data-set the changes of getting a correct output from the modelled machine is not great .
4. Taking an example : 
      - The data set contains pics of car models then the model will make cluster like sedan,suvs,bonnet shape,car shape etc.
   

#### Procedure : 
1. It is used for three major tasks :   
         
   - Clustering
   - Association (relationship between the cluster of groups )
   - Reduction of dimensionality 
      - Reducing number of features in the data set.
      - Transforming high-dimensional data into a lower-dimensional space that still preserves the essence of the original data.
      - Due to this the data focuses on major features but not on minute and specific features.

2. Clustering :  
   1. It is a data mining technique which groups unlabelled data based on their simalarities and differences.
   2. Clustering algos are used in raw processes, unclassified data objects to classified, structured data objects.
   3. Clustering algos are catergorized into : 
         
         - Specially Exclusive
         - Overlapping
         - Hierarchial 
         - Probalistic

### Reinforcement Learning :

#### Defination : 
<em>

Reinforcement is a machine training method based on rewarding desired behaviours and punishing undesired behaviours.

An agent is learning by exploration in an effective environment.

</em>

#### Procedure :

1. Here Developers build/create a method for rewarding for desired behaviours and negative punishment for undesired behaviours.

2. The Markov decision is the basis of the reinforcement systems.

3. In this an agent exists in a specific state inside an environment.

4. The Agent shoud choose the best possible action from multiple potential actions which it can perform in it's current state.

#### Why it is not a famous method to use ?

1. It is difficult to deploy and remains limited in application.
2. Not very reliable systems.
3. With the reinforcement learning problem however, it's difficult to consistently take the best actions in a real-world environment because of how frequently the environment changes.

4. Supervised learning can deliver faster, more efficient results than reinforcement learning.

#### Common Algorithms : 

1. State-action-reward-state-action :

```
- It is policy based algorithm.

- Here the agent is given an policy/rule book.

- Determining the optimal policy-based approach requires looking at the probability of certain actions resulting in rewards or punishments to guide its decision-making.

```
2. Q - Learning : 
```
- In this it takes an opposite approach , here the agent is not provided with any policy/rule book but it learns from expolorations/actions.

- Self-Directed Model.

```
3. Deep Q Model : 
```
- Combination of Q networks and neural networks .

- The network based future actions on a random sample of past beneficial/punishment actions.

```
---

## Machine Learning 

### - Machine Learning Model : 


- Inputs are called as features or factors.
- Output is the predictive value which is based on the inputs(the relation between dependent and independent variables).

### Features :
<em> There are different kinds of features </em>

1. Qualitative : categorical data ( finite number of categories or groups ).
   ### Nominal Data:
   1. In this data there is no <b> Inherent Order </b>, basically a <b> Nominal Data Set</b> .

   2. For Feeding into our system we need to use <b> One Hot Encoding </b> Methodology.

      - In this if the data element belongs to one category then it is set to 1 else it is set to 0.

      - Example : 
            5 males & 2 females:  
               - MALE   [1,0,0,1,1]  
               - FEMALE [0,1,1,0,0]

      Eg :  Gender (m/f) , Branches in the universities ,etc.

   ### Ordinal Data :
   1. In this data there is <b> Inherent Order </b>, basically a <b> Ordinal Data Set</b> .

   2. For feeding into our systems we can <b> Ranking Methodology</b> or <b> Weights Methodology </b>.
   
   Eg :  age groups , price groups , brand ranking etc.

---

## Supervised Learning Tasks :

### 1. Classification ( Predict Discrete Classes )

   #### - Multi-Class Classification 
         - Cat/Dog/Rabbit/Etc Multiple Objects/Entities
         - Orange/Apple/Banana/Etc 
         - Plant Spices 
         
   #### - Binary-Class Classification
         
         - Positive/Negative Sentiments 
         - Cat/Dog  Objects/Entities
         - Spam/Not Spam 

### 2. Regression ( Predict Continous Values )

      1. Examples :  Stock price tommorrow , Temperature tommorrow,etc.

---
### How to Train the Model / How To Find the Model Performance 


- The Data Set is Quantitative data as it is scaled from 1-10.(Ordinal Data)

- Here we have a ouput column which is not given to the machine but it is used as a reference , if the predictive value is same as the output value.

- To Train the Model we break the data-set into 3 sub-data sets 
         
         . Training data set
         . Validating data set
         . Testing data set 
- Training the Model :


- When we use Validation set then the feedback loop is not closed means if there is a loss between the predictive ouput and actual output then adjustments in the model is done.

- Here we come up with different models and find out the least loss occuring model.

- After finding the accurate model then we use testing data set to gain Final Report Performance of the model on the new data .

### Metrics of Performance ( LOSS ):

- Loss : It is the Difference between the Predictive and actual Output values.

- Types of Losses :

#### 1. L1 Loss :


- So it is the Summation of the prior loses and the current loss.

#### 2. L2 Loss :


- So it is the Summation of square of Difference of output values / Loss.

- Parabolic in Nature ( Quadratic )

- If the predictive value is close to actual value then the penalty is minimal and if it far away then the penalty is much higher.

#### Binary Cross Entropy Loss

- Loss = -1 / N * sum(Y.real * log(Y.predicted) +(1 - Y.real) * log(( 1 - Y.predicted )))

- It just say the loss decrease as the performance gets better.

---
## Let's Come to Some Basic Coding Stuff 

### Problem 1 : 

- It is basic code where we are just reading the csv adding column names , making a new column named value which consits the value 1 for gamma and 0 for hydrons.

- Then We make Histrogram for Gamma and Hydrons. which overlap each other.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Making Columns
# 1.  fLength:  continuous  # major axis of ellipse [mm]
#  2.  fWidth:   continuous  # minor axis of ellipse [mm]
#  3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
#  4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
#  5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
#  6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
#  7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm]
#  8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
#  9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
# 10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
# 11.  class:    g,h         # gamma (signal), hadron (background)

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
        "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
# Converting it into a Data Frame
df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)
# Displaying first 5 rows of the data frame
print(df.head())
# It will give the unique objects present in the column
# for this example we will get g -> gamma h-> hydrons
# it can be f-> female or m -> male
print("The Class Column for the Data Set \n" + str(df['class'].unique()))
#  How to check if the whole column contains certain values ?
df['value'] = (df['class'] == 'g').astype(int)
# If the whole column has only g then it will give 1 or if contains h then it will return 0
print(df)

# Here we are taking all the columns but not the output column.
for label in cols[:-1]:
    # So for each label only the value == 1 rows will come and it will make a histrogram.
    plt.hist(df[df["value"] == 1][label],
             color='blue', density=True, alpha=0.7, label="Gamma",)
    plt.hist(df[df["value"] == 0][label],
             color='red', density=True, alpha=0.5, label="Hydrons",)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    # Overlapping
    # The legend function in Python is used to place a legend on the axis of a plot or subplot.3 The function has an attribute called loc that can be used to specify the location of the legend.
    plt.legend()
    plt.show()

```


### Problem 2 :

- It is basic code where we are just reading the csv adding column names , making a new column named value which consits the value 1 for gamma and 0 for hydrons.

- We are dividing the data set into 3 catergories   

   ### 1. Training Data Set
   ### 2. Validation Data Set
   ### 3. Testing Data Set
   
- Dividing it into such a way that the data set is used fully before model deployment and all the divisions should be made on the random basis so that all types of data can be used in training phase.

- Then we made a Scaling() which scale the data set (here the data set with less amount we just add random data from it's same data set.)
 

```py

# Following the model Training process
# Training data set , Validation data set , Testing data set
import pandas as pd
import matplotlib as plt
import numpy as np
# class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
# std:standard Score
# z = (x - u) / s
# x is actual value
# s is standard deviation
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)


df['value'] = (df['class'] == 'g').astype(int)

print(df["value"].head())

train, valid, test = np.split(df.sample(frac=1), [(int)(
    0.6 * len(df)), (int)(0.8 * len(df))])

# print(train)
# print(valid)
# print(test)

# # Here when we see the data set the values in the colums differs like one column has 100+ values and one column has value in 0.0+


def scaling(df, oversample=False):
    x = df[df.columns[:-2]].values  # All columns values
    y = df[df.columns[-1]].values  # Last col values

    # Creating a Scaler
    scaler = StandardScaler()
#     # from scaler we can do fit and transform x
    x = scaler.fit_transform(x)
    if oversample:
        ros = RandomOverSampler()
        # by this function it will take the low sample and add some random sample from the same data set.
        x, y = ros.fit_resample(x, y)
#     # Now the whole modified data will be created using 2D numpy
#     # hstack is horizontal stacks of arrays
#     # We will be modifying the y because x is a 2D matrix whereas y is array so making it into a 2D matrix
    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y


# Calling the Scaling Function for Equallising the training data set.
Train, x_train, y_train = scaling(train, oversample=True)
# Here Oversample is False because these are Testing and validating data.
# We just want the Accuracy of the Model which can be analysed by any length of the data set
Valid, x_valid, y_valid = scaling(train, oversample=False)
Test, x_test, y_test = scaling(train, oversample=False)
print(y_train)
# The Gamma Data Set
print(np.sum(y_train == 1))
# The Hetrons Data Set
print(np.sum(y_train == 0))
# Here we will get same size due to Random Over Sampler

# From this Program we can see that all our data

```

----
<br>

## K - Nearnest Neighbours

- Taking  an example if a person has a car or not and plot the graph on the basis of income and no. of kids.


- So here 1st Observation is that it is binary classification.

- All the Sample(data) has a lable (+/-)

- So just take if a person has $40k income and has 2 kids then our model should say he can't have a car because the there in no nearby point to justify.

- Here if we can see we can use graph method for calculating distances .

- The Graph must be bi-directional and weigthed.

- But Graph may arise problem in multiple paths between the edges and it won't be suitable for large scale.

- So we Use <b> EUCLIDEAN DISTANCE </b> it is just a straight line distance between the edges.

- Formula : 

   ### distance = √ (x1-x2)² + (y1-y2)² 

#### - Here k means how many number of edges/neighours we have to take for any  judgement. (Generally 3 or 5)

### How is the Eucidean Method is Used : 

- Taking above Example of Man :
<br>
<br>

#### Analysis : 
   ** if K Value is 3   

   1. Nearest Neibours all are blue :  so the man don't have car
<br>

---

<br>
- Taking another Example :
<br>
<br>


#### Analysis : 
   ** if K Value is 3   

   1. Nearest Neibours all are :

         - One Blue 
         - Two Red  
      So the Man/Person have car.

---

### Coming to Coding Stuff : 

- First we imported Library named :   
<em> " sklearn.neighbors from KNeighbotsClassifier "</em>
  
  Here we are using already model from the library.(to avoid human errors and bugs).

- Then normal code that we did in above programs and algorithms.

- Then made a model (kn_model) by using KNeighborsClassifier(K value :  number of points to take as reference)

- Then Trained the model by using fit(values of training)

- Then Took a Prediction test by using x_test data set to get predictive y values , so that we can compare both predictive y values and actual y values.

- Here we have 2 methods to check the accuracy of the model

         - By Libraray Function : 
                  - for this we have to import a library "sklearn.metrics import from classification_report"

                  - Then we call classification_report(actual values , predictive values)

         - By My Own Function : 
                  - for this i created my own comparison function named : accuracy(actual values,predictive values)

                  - In the i am calculating the miss match count between the both.

                  - For accuracy can be calculated by : 100 - miss_percentage

- By Calculating the excution of time by both of the methods my method took 0.7 ms whereas library function took 11.20 ms but it was feasible because the data-set was small.


```python 

# Coding K-nearest Algorithm
import time
import pandas as pd
import numpy as np
# The Lib Directly provides Function for use.
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)


df['value'] = (df['class'] == 'g').astype(int)

print(df["value"].head())

train, valid, test = np.split(
    df.sample(frac=1), [(int)(0.6*len(df)), int(0.8*len(df))])

# print("\n The Training Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(train)
# print("\n The Validating Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(valid)
# print("\n The Testing Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(test)


def accuracy(y_test, y_pred) -> int:
    miss = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            miss += 1

    return miss


def scaling(df, oversampling=False):
    x = df[df.columns[:-2]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversampling:

        ros = RandomOverSampler()

        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y


Train, x_train, y_train = scaling(train, oversampling=True)
Valid, x_valid, y_valid = scaling(valid, oversampling=False)
Test, x_test, y_test = scaling(test, oversampling=False)
print("\n\n\n TRAINING : \n\n\n")
print(np.sum(y_train == 1))
print(np.sum(y_train == 0))
print("\n\n\n VALIDATION : \n\n\n")
print(np.sum(y_valid == 1))
print(np.sum(y_valid == 0))
print("\n\n\n TESTING : \n\n\n")
print(np.sum(y_test == 1))
print(np.sum(y_test == 0))

# From now we will code K-Nearnest Algo

kn_model = KNeighborsClassifier(n_neighbors=2)
# Taking k value as 2
# now i am calling the fit function which will basically train or set the data in the model
kn_model.fit(x_train, y_train)
# Now i will call predict() function where i will pass the test data set to test the model.
y_pred = kn_model.predict(x_test)

print("The Original Y Test Values : ")
print(y_test)
print("The Predicted  Y Test Values : ")
print(y_pred)
# Just made a function for checking the efficiency of the program
print("\n\n\nTesting the Efficiency of the Model :\n\n\n")
st1 = time.time()
miss_values = accuracy(y_test, y_pred)
miss_percent = ((miss_values)/(len(y_test)))*100
accuracy_percent = 100-miss_percent
print("\n\n\n-------- OWN FUNCTION -----------\n\n\n")
print(
    f"The Accuracy is : {accuracy_percent} % \nThe Miss Percentage is : {miss_percent} \nNumber of Misses are : {miss_values} / {len(y_pred)}\n")
et1 = time.time()

# The Report that skLearn provides (Library function)
st2 = time.time()
print("\n\n\n-------- LIBRARAY FUNCTION -----------\n\n\n")
print(classification_report(y_test, y_pred))
et2 = time.time()

# Here i am comparing the Time Taken for both of the functions .
print("The Execution Time of\n| OWN FUNCTION  \t| LIBRARY FUNCTION \t|\n")
print(f"| {(et1-st1)*10**3 } ms | {(et2-st2)*10**3 } ms |\n\n")

```

---
<br>


## Naive Bayes Algorithm

### Let's Understand Bayes Rule :

        P(ck/x) = P(x/ck) . P(ck)
                 -----------------
                        P(x)
        
        - P(ck,x) -> Probability (posterior,feature-vector)
        - P(x/ck) -> Probability (LIKELYHOOD)
        - P(ck)   -> Probability of Prior
        - P(x)    -> Probability of Evidence

### yˆ = argmax P(ck) * ¶ P(Xi/Ck) k -> {1,...k}

### MAP : Maximum A Posteriors 

- We can use acf or pacf charts/graph to find the MAP estimates.

- To find the MAP estimate of X given that we have observed Y=y , we find the value of x that maximizes.

### Working of Naive Bayes

- Convert the data-set into frequenc tables.

- Generate Likelihood table by finding by finding the probabilities of features.

- Then use Bayes theorem to calculate the posterior probabilities.

### Advantages of Using Naive Bayes Theorem 

- It is an easy and fast ML algorithm.

- It can be used for Binary-Class as well as Multi-Class of Data-Sets.

- The Performance of Naive bayes is very attractive for Multi-Class Data-set.

### Disadvantage of Using Naive Bayes Theorem :

- The only disadvantage that Naive bayes is that it considers all the factors/variables are independent and there is no correaltion between the variables.

### Applications of Naive Bayes Theorem :

- It is used in Evaluation of Credit-Scoring.

- It also used in Medical data Classification.

- It is largerly used in text-analysis means in sentimental analysis & spam filtering.

### Types of Naive Bayes Models : 

1. Guassian Model :
    
    In this model we have to assume that features follow a Normal Distribution.

    It takes continous values and also predict continous values rather than discrete values.

<br>

2. Multinominal Model : 
    
    This model is used when the data is multinominal distributed.
    
    It is generally used in data where we have to classify the data into different categories.

    Like the documents/articles that needs to classified into categories like sports,politics,education,entertainment,etc.

    Here the model uses frequecy of words.

<br>

3. Bernoulli Model : 

    It is similar to multinominal model but here the variables are independent Boolean variables.

    Like if a particular word/phrase is present in the document/documents.

### Now Come to the Coding Stuff : 

```python

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

```

<br>

--- 

## Logistic Regression 

- It is a Statistical model which is used for classification and predictive analysis.

- Logistic regression can be used to classify an observation into one of two classes
(like ‘positive sentiment’ and ‘negative sentiment’), or into one of many classes.

- Logistic Regression estimates the probability of having an event.

- The outcome will be a probability.

- The Logistic Regression was made because of linear regression irregularities.

- In Linear Regression the Best-Fit Line dosen't cover all the data points and contains error terms.

- So due to this logistic regression was made basically here we classify the data into categories using Regression.

- If we have only one feature x then it is : SIMPLE LOGISTIC REGRESSION

- If we have x1,x2,x3 ... xn then it is : MULTIPLE LOGISTIC REGRESSION

<br>

As the best fit line values goes from [-inf,inf] so we futher use probabilty function as the value will be in range of [0,1].

<br>

<br>

So we use sigmoid function where we fit our data . below there is sigmoid garph.


<br>
 
Here I want Numerator as 1 

<br>

- Now the Equation resembles like Sigmoid Function which is used in binary decision making.

<br>


- Formulas : 
    
    - Logit(pi) = 1/(1+ exp(-pi))

    - S(mx + b) => Sigmoid Function 


### Difference between Naive Bayes theorem and Logistic Regression :

| Naive Bayes | Logistic Regression |
| --- | --- |
| It is a Generative Classifier | It is a Discriminative Classifier |
| In this Model the train itself by the features of the general data | In this Model it trains itself by classifiers that is present in the data.|
|Taking  an example : Here we have to explicity ask the model to generate i.e draw a dog/car/chair etc.| Here the Model trys to learn from the data like if data has dogs,cats,stairs so from here it will learn like all the dogs have belts,stairs are in scalene triangle.|

---

### Uses of Logistic Regression :

1. Fraud Detection : Logistic regression models can help to find data anomalies,some characteristics may have higher association with fraudulent activities.

2. Diease Predictions : Logistic regression models can help to find body's data anomalies.can find the expected dieases that can happen.The model can also be used for specific diease as binary model (O/P : True or False).

3. Churn prediction : here the model can predict the probabilitie of high performer of leaving the organisation.

### Coming to Coding Stuff : 

```python

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

```

<br>

----

## Support Vector Machine

- It is a powerful ML algorithm used for linear,non-linear classification,regression.

- It works better for smaller data sets.

- It might not be the best model where data has outlayers.

- The aim for SVM is to create the best line or decision boundary that can segregate n-data points basically make classes so that it can easily put new data into correct class.

- The Best Decision Boundary is called Hyperplane.

- SVM chooses extreme points/vectors which helps to create Hyperplane and the extra vectors/points are called as support vectors.

<br>



### Types of SVM :

1. Linear SVM : In this the data set can be seperated into 2 categories/classes using a straight line.

<br>

2. Non-Linear SVM : Here the data set cannot be seperated by  a linear straight line.so we use different methods/ways to seperate the classes.

<br>

** There can be multiple method this is one of the ways/methods.  
** There can be Multiple Straight lines not one is compulsory.

### Coming to the Coding Stuff : 


```python

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
   
   print(f"\nThe Time Taken by the Model is : {end1-st1:.6f} \n")
    
main()

```

<br>

### ** Here the Accuracy of the Model is Awsome so far better than other models. Giving around 87% - 90% which is pretty good only thing it takes alot of time .

<br>

--- 

<br>

## Neural Networking :

1. Neural Network is basically a method/model in AI that teaches the computers to process the data like human brain (basicaly find connections between the data points,analyse the data points etc)

2. The Neural Network is divided into 3 Layers : 
    - Input Layer :
        
        - here we input the data to the neural network , Here the Input data is processed,analysed by the nodes and then passed into next Layer.

    - Hidden Layer :

        - Here it takes input from input layer or other hidden-layers.Each hidden layer analyse the output from previous layer process it and pass to next layer.

    - Output Layer : 

        - This layer gives all the processed data value as ouputs,It can be single or multiple nodes.If the problem/data was of binary like true or false OR 1 or 0 then we one node as answer, but if the problem/data involves multiple class/variables/factors then we will have multiple nodes in the output layer. 

<br>

<br>

3. It is generally used in summarization and generalisation.

4. The Model structure is : Input data , weights , Neurons , bias/Noise terms , Activation Function , Output data .

5. Here we have weights attached to each data point which is multiple and summed up and passed to Neuron .

    #### EQUATION : 
            𝓍𝟢 * 𝒲𝟢 + 𝓍𝟣 * 𝒲𝟣 + 𝓍𝟤 * 𝒲𝟤 .... 𝓍𝓃 * 𝒲𝓃

### Types of Neural Networks :

1. Feed-Forward Neural Network : 
    
    - Feed-Forward neural network process data in one direction from input node to output node.

    - .Every node in one layer is connected to every node in the next layer.

    - This Neural network uses feedback mechanism to improve the predictions.

<br>

<br>

2. BackPropagation Algorithm : 

    - The Neural Network continously learn using the feedback mechanism to imporove the predictive analysis.

    - In a Neural Network there a multiple paths to reach the output layer but to find the most optimal,correct path so that our prediction values are imporved so we use feedback loops inside the network.

    - It works in 3 stages : 
        
        - Each node makes a guess about the next node in the path (can be called as recursive permutation tree).

        - Validate if the guess is correct ? here it follows a assumption that if higher weight of the node means going to a correct path.

        - For next data point it will make same prediction using same assumption on weights.

    - The Equation for the backpropagation is :

            𝓦0.new_value = 𝓦0.old_value + α *
        
        
        - Here the addition sign is due to the Negative gradient/going dowards.

        - Take Small steps and not large steps and check if we are going towards 0/x-axis.

        - α :  is the Learning Grade means how long/steps will be required so that our Neural Network will converge or can diverege.


<br>

<br>

```
    - The Graph of Backpropagation is : 
```

<br>

<br>

```

        - it is basically a L2 loss graph.
        - The closser decesend gradient goes to 0 our number of back-steps decreases.
        - Here we can find the value using descend gradient function.

```
<br>

<br>


- Convolutional Neural Network : 

    - The hidden layer in the Neural Network perform specific mathematical,statsitical functions/methods.

    - Convolution is summarizing or generalizing or filtering process.

    - Convolution is useful in Image processing/classification as they can extract relevant pictorial features from the images.



### Train a NN (Neural Network) : 

1. The Intial Step is to train the Neural Network using a labelled or unlabelled data (Supervised learning or Unsupervised learning)

2. Making a weighted product data to pass neutron node/nuteron nodes.

3. As well as we pass a bais term/error term to the neutron node.

<br>

<br>

4. Then we pass this node to a Activation Function (statistic,mathematical function/method).

5. The Output of Activation function is the Final " OUTPUT ".

<br>


<br>

- Here the training data is passed to the Model which gives ouput .

- Then the output is compared with the test data and the deviation list/array/values are calculated/evaluated.

- Then we pass the deviation values again to model so that it can reduce them and provide more accurate predictions . 

<br>


<br>

### What is the Activation Function ?

- Without an Activation Function the Neural Network becomes a Linear Combination Weight Summation Model.

- Activation Functions are not linear :
<br>


<br>

<em> They are like these : </em>

- Whenever the partial - processed data moves from one node/layer to another then it will have some non-linear terms which is introduced by the Activation function .. so that it just dosn't become a linear summation of combinations.


#### Libraries for NN ( Neural Network ) : 

- So Developing a Neural Network Model we use Tensor-Flow.

- It is an Open-Source Library which helps us to develop ML models.

<br>

### Comming to Coding Stuff ..  developing  a Neural Network for Classification :

```python

# Developing a Neural Network
# This is very naive code I have to optimise this using GPU and threads for this.

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

```

<br>


### Understanding Tensor Flow and How to Build a Neural Net ?

- Keras is an high-level API of the Tensor Flow platform.

- It provides an interface for solving high level ML algorithms/problems.

- Layers :

   - tf.keras.layers.Layer class which is fundamental abstraction keras.

   - A layer encapsulates a state (weights) and some computation call tf.keras.layers.Layer.call

   - Here the layers handle the preprocessing of the data,tasks like Normalization and text vectorization

- Models :

   - Model is an object which group the layers and that can be trained on the data.

   - The Simplest Model is  <b> " SEQUENTIAL MODEL " </b> which has linear stack of layers.

      - A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

      - A Sequential model is not appropriate when:

         - Your model has multiple inputs or multiple outputs

         - Any of your layers has multiple inputs or multiple outputs
         - You need to do layer sharing

         - You want non-linear topology (e.g. a residual connection, a multi-branch model)

      - Here We can implement this by two ways :

      Method 1

      ```python
         
         model = tf.keras.Sequential(
            [
               tf.keras.layers.Dense(units = x,activation = "relu",input_reshape = (y))

               tf.keras.layers.Dense(units = x,activation = "relu",input_reshape = (y))

               tf.keras.layers.Dense(units = 1,activation = "sigmoid")

            ]
         )
         
      ``` 
      Method 2
      ```python

         model = tf.keras.Sequential()

         model.add(tf.keras.layers.Dense(units = x ,activation = "relu",input_reshape = (y)))
         
         model.add(tf.keras.layers.Dense(units = x ,activation = "relu",input_reshape = (y)))
         
         model.add(tf.keras.layers.Dense(units = 1 ,activation = "sigmoid"))

         model.pop()

         print(len(model.layers))
      
      ```

      -  Once the Sqequential model is trained and developed then it behaves like a functional api model.

      - So here every layer has an input and an output,which means adding a feature extractor which extracts the features from each layer at each epoach. 
      
      - The Code for that is : 

      ```python

         def feature_extractor(model):
            extractor = tf.keras.Model(
               inputs = model.inputs,
               outputs = [layer.output for layer in model.layers],
            )
            
            return extractor

         # calling in the main function in the for loop and add this extractor in an array 
         # It will contain each layer's inputs and outputs.
      
      ```
   #### - IMPORTANT CONCEPT is Transfer Learning with Sequential Model :

   - There aere two methods to transfer learning from one layer to another.

   - Here we freeze all the layers expect the last one .

   - Code Implementation :

   #### Method 1 :

      ```python

         model = tf.keras.Sequential(
            tf.keras.layers.Dense(units = x,activation = "relu",input_reshape = (y))

            tf.keras.layers.Dense(units = x,activation = "relu",input_reshape = (y))

            tf.keras.layers.Dense(units = 1,activation = "sigmoid")
         )

         model.load_weights()

         # Expect the last layer   
         for layer in model.layers[:-1]:
            layer.trainable = False

         
         model.compile()
         model.fit()

      ```

      #### Method 2 :

      ```python
      
      # Load a convolutional base with pre-trained weights

      model = tf.keras.applications.Xception(
         weights = "imagenet",'
         include_top = False,
         pooling = "avg",
      )

      model.trainable = False

      model.complile()
      model.fit()
      
      ```
---

<br>

   ### Complex Models like : 

   - <b>" FUNCTIONAL API " </b>  in keras,it is more flexible model which allows to add more layers in the model.
         
      - It can have multiple inputs and outputs and also allows to share between them.
      -  It is a data structure and it is easy to save it in single file and can use it in multiple other files.
   
      - The Model creates a Directed Acylic graphs (DAGS) of layers which are inter-connected.


      - Here the model class offers a build-in-training loop (the fit Method() ) and build-in-evaluation loop (the evaluate Method() )

      - We can also use this in customized looping Machine learning algorithms like Gan's etc.

      - Save & Serialze of the Model Process :

         - Saving the model and serializing the work for building model using functional api as we do in sequential model.

         - The way we save the functional model state by using the call function <b><em>" model.save() " </em></b> to save the entire model in single file.( By this we can recreate and reuse the model in various files)

      QUESTION What the saved file contains ?
      - It contains these things/components :

         - Model Architechture

         - Model Weights Values

         - Model Training Configurations (the parameters passed at the time of compilation of the model)

         - Optimizer and it's state (the state of the model)

      - Code Implementation of the Save Function :

         ```python

            model.save("path_to_my_model.keras")
            del model
            # Recreate the exact same model purely from the file:
            model = tf.keras.models.load_model("path_to_my_model.keras")

         ```
<br>  

---

<br>

   - <b>" SubClassing "</b> in keras,as sequential model dosen't provide any flexibiliy whereas functional api model provides a little bit of flexibility but to make a model with scratch which can only be done using a call method.

   - Now just take a look on the functionalities on the Models : 
      
      - tf.keras.Model.fit: Trains the model for a fixed number of epochs.

      - tf.keras.Model.predict: Generates output predictions for the input samples.

      - tf.keras.Model.evaluate: Returns the loss and metrics values for the model; configured via the tf.keras.Model.compile method.

<br>

----

<br>

## Linear Regression ( One of The Important Topic to Focus on ! )

- Linear Regression is one of the model which checks the relationship between the variables.

- So there is an dependent variable and there are one or many independent variables .

- The Linear Regression Equation is :

      y  =  β1.x1 + β2.x2 + β3.x3 + .... βn.xn + µ

   - y  =>  Is the Dependent Variable/Regressand.

   - x1,x2,x3 ... xn  =>  Is the Independent Variables/Regressor. 

   - β1,β2,β3 ... βn  =>  Is the Coefficient which states the magnitude of the relation between the specific variable and dependent variable .

   - µ  =>  Is the Error Term.

   like example :

      equation like this : y  = 0.32.x1 + 0.87.x2 + 0.5.x3 + µ

      So analysis of this equation is :

         - y => Is the independent variable.
         
         - 0.32 times the x1 value effects the y variable.

         - 0.87 times the x2 value effects the y variable.

         - 0.5/half times the x3 value effects the y variable.

         - µ => Error Term 

- Here we have to create a best-fit line which intersects most of the data-points and other data-points are not so much deviated.

- Here we will have to make a model where we evaluate the model performance using the best-fit line and the calculating the error terms from it.

   - First we make/create a best-fit line and find the error terms/deviation terms then we sum up these terms

   - And Our main Goal is to reduce the summation of the error terms.


-  Using OLS METHOD (Ordinary Least Squares) instead of Sum of residuals.

#### OLS Method

- It is basically used to validate the assumptions of linear regression model.

- Most common method to estimate the coefficients of a linear regression model.

- The Main objective of OLS is to minimize the sum of the squared differences between the actual values and predicted values.

- Stating the Equation :

## &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;∑ ( 𝓨𝓲 − 𝓨𝓸 ) ^2

### . 𝓨𝓲 -> actual value

### . 𝓨𝓸 -> predicted value

---

### Why Square the Residuals?

1. If we don't do then the negative and positive residuals might cancel each other out which leads to many points to lie in the best-fit-line.

2. Squaring ensures all are positive, then we can focus on minimizing actual prediction errors.

---


### ANALYSIS OF THE PLOT

```

1.Approximate we can see a positive linear relationship between x and y which means that increases in x we can see an increase in y also.

2. We can draw a best fit line which will intersects the maximum data-points.

3.The vertical distances from the best-fit-line is called deviation or error-term.

```

### Adding the Best-Fit-Line to the Scattered Data Point Graph

### Analysis of Best-Fit-Line Plotted Graph :

```

1. Finding the Best Line: The "best" line is determined by values of α (intercept) and β (slope) that minimize the RSS.

```

### Using the OLS METHOD TO VALIDATE THE MODEL'S OUTPUT :

### Analysis of OLS Plotted Graph :

```
1.The primary goal of OLS is to find a line (or a "fitted line") that best fits a set of data points.

2. '𝓨𝓲' is actual value and '𝓨𝓸' is predicted value and difference between them is called " 𝓻𝓮𝓼𝓲𝓭𝓾𝓪𝓵 " (residual) .

3. Here we will find RSS (Residual Sum of Squares) : This is the sum of all the squared residuals.It indicates the errors our model has proposed.

```

### Analysing the Linear Regression Model using Mean Absolute Error (MAE) :

- Here we sum of the residuals and then take the mean of them.

- The Mathematical Equation is : 

```
      𝓷
      Σ  | 𝔂𝓲 -  𝔂^ |
      𝓲
      ------------------
            𝓷
```

- It states that this is the average difference between the predictive value and actual values.

### Analysing the Linear Regression Model using Mean Squared Absolute Error (MSE) :

- It is the squaring of the difference between the predicted value and actual value and adding them.

- It helps to punish large errors in the predictions and prevent the model from predicting values with large deviations.(helps for covering range of errors by squaring them).

- The Mathematical Equation is :

```
      𝓷
      Σ  | 𝔂𝓲 -  𝔂^ |^2
      𝓲
      ------------------
            𝓷
```

- Sometimes the Y variable is impossible entity to be squared off then we use Root Mean Squared Error (RMSE) methods.

### Analysing the Linear Regression Model using Root Mean Squared Error (RMSE) :

- So it solves the problem which is faced by when using MSE method,the impact on the y values and the values we were getting.


- So here we add the deviation values and then square it and evaluate/calculate the summation of those terms.

- After summation we root the value to get RMSE value.

- The Mathematical Equation is :

```
    _______________________
   |
   |   𝓷
   |   Σ  | 𝔂𝓲 -  𝔂^ |^2
   |   𝓲
   |   ------------------
  \|         𝓷

```

- Here the value of RMSE is that we get the same unit of value as Y variable is.

- if Y is in meters then  

   - MSE value will be in meter ^ 2

   - But RMSE value will be in meters


<br>

---

<br>

## Linearity and possible forms of regression function :

### -> So for using OLS the model should be linear that means the relation between ' 𝔁 ' and ' 𝔂 ' should be described through a straight line.

### -> Here the parameters should be also linear.(α,β)

## \*\* That means α,β should not be multiplied,squared,cubed,etc.

## Exponential Regression Model :

### -> In Exponential regression model since Y varies according to some exponent (power) function of X.

### -> Here we need to convert this exponential equation into a linear equation so we have to use log both sides and simply the equation and then we can use OLS(Ordinary Least Square).

```

               Y = A * X^β * e^ut

```

---

## Assumptions in Linear Regression Model :

| Technical notation | Interpretation                                                                      |
| ------------------ | ----------------------------------------------------------------------------------- |
| E(ut ) = 0         | The errors have zero mean                                                           |
| var(ut ) = σ 2 < ∞ | The variance of the errors is constant and finite over all values of xt             |
| cov(ui , u j ) = 0 | The errors are linearly independent of one another using Covariance function        |
| cov(ut , xt ) = 0  | There is no relationship between the error and x variable using Covariance function |

---

## HYPOTHESIS TESTING :

### -> Hypothesis testing is a statistical method which is used to draw conclusions for the financial theories and the analysis model.

### -> It is a Structured way to determine whether there is enough data to support an assumption.

### -> There are 2 Categories of Hypothesis :

### - Null Hypothesis

```
** Here we say there is no relation between the variables .

1. It is a Statement which has default status and zero happening.

2. So we assume that the assumption is true until we find any proof against the statement.

3. It is a quantitative analysis.

4. The Conclusion statement if the Null hypothesis fails will be :

" The given set of data does not provide strong evidence against the null hypothesis because of insufficient evidence ."

```

### - Alternative Hypothesis :

```
** It test over the Relationship between the two variables.

1. Here we assume a statement that there is a relation between the variables and then test to prove this assumption.

2. Like assuming that x is inversely proportional to y and z then we have to use a data set to prove this statement.

3. The Conclusion statement if the Alternative hypothesis fails will be :

" The given set of data does not provide strong evidence against the relationship between the x and y variables because of insufficient evidence ."

```

---

## The Confidence Level While doing Regression :

### While doing Null Testing & Alternative Testing and finding out the confidence range from the given sample data.

### The Plots to Explain the Analysis Results :

### Here the ' 𝓑^ ' -> Standard errors away from the best fit line.

### Here the ' 𝓑\* ' -> Value obtained from Null hypothesis testing.

### Determining the Rejection by Certain Conditions.

### \*\*\* 𝐻𝑜 is Null Hypothesis Testing O/P or VALUE.

### \*\*\* 𝐻𝟣 is Alternative Hypothesis Testing O/P or VALUE.

---

### Condition 1 : if 𝐻𝑜 : 𝓑 = 𝓑\* && 𝐻𝟣 : 𝓑 != 𝓑\*


---

### Condition 2 : if 𝐻𝑜 : 𝓑 = 𝓑\* && 𝐻𝟣 : 𝓑 < 𝓑\*

---

### Condition 3 : if 𝐻𝑜 : 𝓑 = 𝓑\* && 𝐻𝟣 : 𝓑 > 𝓑\*


---

## A special type of hypothesis test: the t-ratio :

### 1. To determine predictions we perform a hypothesis test on our developed method/model.

### 2. The t-ratio is a measure used to determine how many standard errors a coefficient is away from zero (or any other value we want to test against, but in this case, it's zero).

### 3. Equation :

```

               t−ratio =  (β^ - β*)
                         ------------
                           SE( β^ )

```

### 4. Terms

- (β^ - β\*) -> difference between the estimated coefficient and testing coefficient (generally zero) .

- SE( β^ ) -> Standard Error of the Coefficient.

---

## Analysis by T-Ratio :

1. The t-ratio will tell us if the change is statistically significant or if it's likely just due to random chance .

2. A large t-ratio means that it's less likely that our observed relationship between two variables is due to random fluctuations .

3. By using the t-ratio, we can determine if financial factors have a real and statistically significant impact on other financial metrics .

---

## Explain, with the use of equations, the difference between the sample regression function and the population regression function.

### 1.Population Regression Function (PRF) :

```

1. PRF function represents true relationship between the two variables where one is dependent and another is independent .

2. EQUATION :

    --- --- --- --- --- --- ---
   |      Y = α + βX + u       |
    --- --- --- --- --- --- ---

3. Terms :

         . Y -> Dependent Variable
         . X -> Independent Variable
         . β -> Slope Coefficient
         . u -> ERROR TERM
         . α -> INITIAL EXPECTED VALUE

```

### 2. Sample Regression Function (SRF) :

```

1. SRF is a estimation for PRF function in this we just take sample data set to check the relation between the variables .

2. EQUATION :

         --- --- --- --- --- --- ---
        |    Y^ = α^ + β^X + u^     |
         --- --- --- --- --- --- ---

3. TERMS :

      . Y^ -> Predicted value of Dependent Variable
      . α^ -> Estimated Initial Expected Value
      . β^ -> Estimated slope coefficient
      . u^ -> Estimated Error term
      . X -> Independent Variable

```

---

## Generalizing the simple model to multiple linear regression

### -> Bivariate Equation for Regression Model.

### -> But There are not just one factor of influence but there are multiple factors.

### Taking an Example of Stock the Factors that influences the price of the stock are :

         . inflation
         . Sector of the company
         . products of the company
         . Company's new policies
         . etc

### So by above explanation the Equation for multivariate factors will be :

---

             yt = β1 + β2.x2t + β3.x3t + ··· + βk.xkt + ut , where t =1,2,...,T

---

#### - x2t , x3t , x4t , x5t ... xkt -> Independent Variables .

#### - β1 , β2 , β3 .... βk -> Estimated Coefficients .

#### - ut -> ERROR Term

#### - yt -> Independent Variable

### WE CAN COMPRESS THIS IN SIMPLE LINEAR REGRESSION EQUATION BY USING MATRICES

```

   - y = X.β + u

   - where: y is of dimension T × 1 X is of dimension T × k

   - β is of dimension k × 1 u is of dimension T × 1


```

---

## Testing multivariate hypotheses: the " F - TEST "

### - As t-test was used to test single hypotheses but multiple variables there will be multiple restrictions and multiple assumptions so we use F - Test .

### - F-test framework where two regressions are required, known as the unrestricted and the restricted regressions.

### - The unrestricted regression is the one in which the coefficients are freely and are composed by previous data .

### - The restricted regression is the one in where the coefficients are restricted, i.e. the restrictions are imposed .

### - Thus the F-test in hypothesis testing is also termed restricted least squares.

### - The residual sums of squares from each regression are determined, and the two residual sums of squares are ‘compared’ .

### - EQUATION FOR F - TEST :

```

              F - Ratio  =   RRSS − URSS  ×  T − k
                             ------------    -------
                                 URSS           m

```

- URSS = residual sum of squares from unrestricted regression

- RRSS = residual sum of squares from restricted regression

- m = number of restrictions

- T = number of observations

- k = number of previous data values

### \*\*\* RRSS == URSS only at very extreme circumstances this would be the case when the restriction was already present in the data.

### RELATIONSHIP BETWEEN &nbsp; " T - Test " &nbsp; and &nbsp;" F - Test "

- T-test is used for just one dependent and independent variable , whereas F-test is used for multiple independent variable.

- We can say that T - test is a special case of F - test as we square the T-test value it will be approxly equal to F-test value.

- So T-test value is = Z and F-test Value is = Z^2

### How to Find Restrictions in Hypothesis ?

1. The number of restrictions in a hypothesis can be calculated as the number of equality signs in the null hypothesis.
   Eg:

   ```
         case 1 : β1 + β2 =2                    1  restriction
         case 2 : β2 = 1 and β3 = −1            2 restrictions
         case 3 : β2 = 0 , β3 = 0 and β4 = 0    3 restrictions

   ```

2. If all coefficients are zero, and the null hypothesis isn't rejected, it implies none of the independent variables in the model can explain variations in the dependent variable,so no relation.

---

## Goodness of fit ( R^2 ) :

- A part of Analysis after performing linear regression.

- It is the Coefficient of Determination, which is used as an indicator of the goodness of fit.

- It shows how many points fall on the regression line

- In our example, R^2 is 0.91 (rounded to 2 digits), which is fairy good. It means that 91% of our values fit the regression analysis model. In other words, 91% of the dependent variables (y-values) are explained by the independent variables (x-values). Generally, R Squared of 95% or more is considered a good fit.

- Multiple R : It is the Correlation Coefficient that measures the strength of a linear relationship between two variables.

```
   1 means a strong positive relationship
  -1 means a strong negative relationship
   0 means no relationship at all

```

-
- EQUATION :

      ```
               ESS
        R^2  = ---
               TSS
      ```

- ESS -> Explained Sum of Squares
- TSS -> Total Sum of Squares

### Problems With R^2 :

- R^2 is defined in terms of variance so we cannot compare the R^2 values with different Models.

- In Simple Regression model it forms patterns and clusters of the data points.so for simple or incorrect model R^2 may show a high value because it is analyzing patterns and not the relationships between variables.

---

---

### Implementation through code : 

```python

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


```


<br>

```python

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


```

<br>



## Regression Using a Neural Network Through Code :


<br>

<br>

```python

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

```

<br>


---
---
<br>

## Now Just Understand Some Models Related to Unsupervised Learning 

- Here we have unlabelled data,from this we can learning anything.

- Here we will Learn some clustering related Algorithms and processing.

### 1 . K - Mean Clustering

- Here we form cluster of the data-points to predict the y variable(dependent variable).

- Clustering is the categories of the data-points,it is purely depends on the nature of the data,value of the data.

- Here K is number of cluster that we want to form from the data-source.

<br>


<br>

STEPS TO PERFORM K-MEAN CLUSTERING (Process is Known as EXPECTATION MAXIMIZATION) : 

Step 1 -> Choose " k " Random Points to be Centroids .

Step 2 -> Calculate distance between data - points and centroid.

<br>


<br>

Step 3 -> Then Assign the Data - points to it's closest centroid.

<br>


<br>

Step 4 -> Now we recompute new Centroids then redo the Step 2 & Step 3 Processes.(It goes on iterating until none of the centroid points move from their places.)


### Implementation through Code :

```python

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

```


---

### 2 . Principal Component Analysis

- It is usually used for dimensionality reduction like if the data has a bunch of features (x1,x2,x3 ... xn) then can I reduced that into one significant feature (one dimension).

#### Step-by-Step Explanation of PCA 


- Step 1 : STANDARDIZATION 

   - The aim is to standardize the range of continous variables so that each variable/factor can contribute in equal analysis and can help in clustering the categories according the data-set.

   - In this step we ensure that our data-set has mean value of 0 and a std deviation of 1.


- Step 2 : COVARIANCE MATRIX COMPUTATION 

   - Covariance measures the strength between the variables/factors and the relationship.

      - Positive: As the x1 increases x2 also increases.
      - Negative: As the x1 increases x2 also decreases.
      - Zeros: No direct relation

- Step 3 : Compute Eigenvalues and Eigenvectors of Covariance Matrix to Identify Principal Components

   - MATHEMATICAL EQUATION  
            
         AX = lambda X

         - A is 2D matrix (n*n)
         - X is also a 2D matrix (n*n)
         - lambda is Scalar values also known as eigen - values.
         - Both matrices are also known as eigen - vectors
   
- Why we use PCA ?

   - Dimension Reduction : It is used to reduce the number of variables present in the data-set.

   - Feature Reduction : we can use it for process of selection of most viable features/factors which effects the data-set.

   - Data - Visualization : Here we can form clusters/groups of data - points by finding out centroid points which untimately helps in data visualization.

   - Multi - collinearity : It helps in understanding the underlying relationships in the data and the also analyse the dataset structure.

   - Data - Compression : As PCA helps in clustering/group of data into features or components which helps in reducing the data and also helps in speeding the process.

   - Noise Reduction : As we analyse the data-set very deeply which can help us to find noises which are present in the data-set.


----
